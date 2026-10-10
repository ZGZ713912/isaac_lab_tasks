"""Check a routed export through RMCS's actual C++ ONNX runtime, without ROS."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

CPP = r'''
#include "onnx_runtime.hpp"
#include "policy_model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::invalid_argument("model, input, count and output required");
        const std::size_t count = std::stoull(argv[3]);
        if (!count) throw std::invalid_argument("empty probes");
        auto config = rmcs_rl::OnnxRuntime::Config{
            .model_path = argv[1], .model_type = "mlp", .sequence_length = 5,
            .feature_size = 32, .observation_size = 160, .action_size = 4};
        rmcs_rl::OnnxRuntime runtime(config);
        std::vector<float> inputs(count*160), outputs(count*4);
        std::ifstream stream(argv[2], std::ios::binary);
        stream.read(reinterpret_cast<char*>(inputs.data()), inputs.size()*sizeof(float));
        if (stream.gcount() != static_cast<std::streamsize>(inputs.size()*sizeof(float)))
            throw std::runtime_error("truncated probes");
        for (std::size_t i=0; i<count; ++i) {
            runtime.run(std::span<const float>(inputs.data()+i*160,160),
                        std::span<float>(outputs.data()+i*4,4));
        }
        if (!std::all_of(outputs.begin(), outputs.end(), [](float x){ return std::isfinite(x); }))
            throw std::runtime_error("nonfinite actions");
        auto policy_config = rmcs_rl::PolicyModel::Config{
            .path=argv[1], .model_type="mlp", .sequence_length=5, .feature_size=32,
            .observation_size=160, .action_size=4, .normalization_from_metadata=false};
        rmcs_rl::PolicyModel policy(policy_config);
        std::vector<double> sensor(160), action(4);
        std::string error;
        double wrapper_error=0.;
        for (std::size_t i=0; i<count; ++i) {
            std::copy(inputs.begin()+i*160,inputs.begin()+(i+1)*160,sensor.begin());
            if (!policy.run(sensor,action,error)) throw std::runtime_error(error);
            for (std::size_t j=0; j<4; ++j)
                wrapper_error=std::max(wrapper_error,std::abs(action[j]-outputs[i*4+j]));
        }
        sensor[0]=std::numeric_limits<double>::quiet_NaN();
        const bool invalid_rejected=!policy.run(sensor,action,error) && error.find("non-finite observation")!=std::string::npos;
        const bool shape_rejected=!policy.run(std::span<const double>(sensor.data(),159),action,error)
            && error.find("dimensions")!=std::string::npos;
        auto hex=[](std::uint64_t value) {
            std::ostringstream text;
            text << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
            return text.str();
        };
        const auto& info=policy.info();
        if (!info.layout_hash || !info.model_id || info.obs_signature.empty() || info.actions_signature.empty())
            throw std::runtime_error("policy wrapper did not retain deployment metadata");
        std::ofstream output(argv[4],std::ios::binary);
        output.write(reinterpret_cast<const char*>(outputs.data()),outputs.size()*sizeof(float));
        std::vector<double> timings;
        std::vector<float> result(4);
        for (std::size_t i=0; i<1100; ++i) {
            const auto start = std::chrono::steady_clock::now();
            runtime.run(std::span<const float>(inputs.data()+(i%count)*160,160),result);
            const auto end = std::chrono::steady_clock::now();
            if (i>=100) timings.push_back(std::chrono::duration<double,std::milli>(end-start).count());
        }
        std::sort(timings.begin(),timings.end());
        auto bad = config;
        bad.sequence_length=8; bad.observation_size=256;
        bool history_rejected=false;
        try { rmcs_rl::OnnxRuntime wrong(bad); }
        catch (const std::invalid_argument& error) {
            history_rejected=true;
            std::cerr << "REJECTED_HISTORY: " << error.what() << '\n';
        }
        std::cout << "{\"samples\":" << count << ",\"timed_calls\":" << timings.size()
                  << ",\"p50_ms\":" << timings[500] << ",\"p99_ms\":" << timings[990]
                  << ",\"max_ms\":" << timings.back() << ",\"wrong_history_rejected\":"
                  << (history_rejected ? "true" : "false") << ",\"policy_wrapper_max_error\":" << wrapper_error
                  << ",\"invalid_observation_rejected\":" << (invalid_rejected ? "true" : "false")
                  << ",\"wrong_observation_size_rejected\":" << (shape_rejected ? "true" : "false")
                  << ",\"layout_hash\":\"" << hex(info.layout_hash) << "\",\"model_id\":\""
                  << hex(info.model_id) << "\"}\n";
        return history_rejected && invalid_rejected && shape_rejected && wrapper_error<1.e-7 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
'''


def select_probes(actor, observations, per_expert=32):
    """Cover every branch plus observations closest to the routing threshold."""
    with torch.inference_mode():
        confidence, route = actor.router(observations).softmax(-1).max(-1)
        route = torch.where((confidence >= actor.routing_confidence)
                            & (observations[:, -14:-10].abs().mean(-1) > actor.routing_load_threshold),
                            route, torch.zeros_like(route))
    selected = []
    for expert in range(len(actor.experts)):
        ids = torch.where(route == expert)[0]
        if len(ids) < per_expert:
            raise ValueError("Runtime probes must cover every routed expert")
        selected.extend(ids[torch.linspace(0,len(ids)-1,per_expert).long()].tolist())
    selected.extend(torch.argsort((confidence-actor.routing_confidence).abs())[:32].tolist())
    ids = list(dict.fromkeys(selected))
    return observations[ids], route[ids].tolist()


def check(checkpoint, model, samples, output_dir, rmcs_root):
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("runtime_exporter", ROOT/"scripts/rsl_rl/export_onnx.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    actor = exporter._build_actor(exporter._load_state_dict(str(checkpoint)))[0]
    if not hasattr(actor,"experts"):
        raise ValueError("Native routed checks require the actual routed MLP checkpoint")
    manifest = json.loads(samples.with_suffix(".json").read_text())
    if (manifest["schema"] != "deformable_sensor_replay_v1"
            or manifest["sha256"] != hashlib.sha256(samples.read_bytes()).hexdigest()):
        raise ValueError("Runtime samples differ from their sensor provenance")
    with np.load(samples,allow_pickle=False) as data:
        observations = torch.from_numpy(data["policy_obs"].copy())
    probes, routes = select_probes(actor,observations)
    with torch.inference_mode():
        expected = actor(probes).numpy()
    output_dir.mkdir(parents=True,exist_ok=False)
    source = output_dir/"runtime_probe.cpp"
    source.write_text(CPP)
    inputs = output_dir/"inputs.f32"
    probes.numpy().tofile(inputs)
    runtime_dir = rmcs_root/"rmcs_ws/src/rmcs_rl/src/policy"
    ort_dir = rmcs_root/"rmcs_ws/build/rmcs_rl/onnxruntime-linux-x64-1.20.0"
    library = ort_dir/"lib/libonnxruntime.so.1.20.0"
    executable = output_dir/"runtime_probe"
    compile_command = ["g++","-std=c++23","-O2",str(source),str(runtime_dir/"onnx_runtime.cpp"),
                       str(runtime_dir/"policy_model.cpp"),"-I"+str(runtime_dir),"-I"+str(ort_dir/"include"),
                       "-I"+str(rmcs_root/"rmcs_ws/src/rmcs_rl/include"),str(library),
                       "-Wl,-rpath,"+str(ort_dir/"lib"),"-o",str(executable)]
    with (output_dir/"compile.log").open("w") as log:
        subprocess.run(compile_command,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = subprocess.run([str(executable),str(model),str(inputs),str(len(probes)),
                             str(output_dir/"outputs.f32")],capture_output=True,text=True,check=True)
    (output_dir/"runtime.log").write_text(result.stdout+result.stderr)
    metrics = json.loads(result.stdout)
    actual = np.fromfile(output_dir/"outputs.f32",dtype=np.float32).reshape(-1,4)
    error = float(np.max(np.abs(actual-expected)))
    if (actual.shape != expected.shape or error > 5.e-6 or not metrics["wrong_history_rejected"]
            or not metrics["invalid_observation_rejected"] or not metrics["wrong_observation_size_rejected"]
            or metrics["policy_wrapper_max_error"] > 1.e-7):
        raise ValueError("Native runtime changed actual sensor actions or accepted wrong history")
    report = dict(status="passed",checkpoint=str(checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  onnx_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  samples_sha256=manifest["sha256"],max_action_error=error,
                  route_counts=np.bincount(routes,minlength=len(actor.experts)).tolist(),
                  native_runtime_sha256=hashlib.sha256((runtime_dir/"onnx_runtime.cpp").read_bytes()).hexdigest(),
                  native_policy_model_sha256=hashlib.sha256((runtime_dir/"policy_model.cpp").read_bytes()).hexdigest(),
                  runtime_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                  native_metrics=metrics,inference_p99_below_10ms=metrics["p99_ms"] < 10.,
                  scope="Actual RMCS C++ OnnxRuntime and PolicyModel on this host; no ROS or hardware motion")
    (output_dir/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--samples",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--rmcs-root",type=Path,default=Path("/home/noir/Documents/workspace/RMCS"))
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint,args.model,args.samples,args.output_dir,args.rmcs_root),indent=2))


if __name__ == "__main__":
    main()
