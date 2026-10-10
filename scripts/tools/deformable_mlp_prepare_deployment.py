"""Prepare an isolated RMCS deployment bundle after qualified long training.

This writes artifacts and a minimal config patch in this workspace. It does not
edit RMCS, launch its processes or command hardware. --wait can accompany the
authorized long-running training process and prepare its final accepted model.
"""

import argparse
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/tools"))
from deformable_mlp_guarded_train import checkpoint_info, qualify, actor_update_info
from deformable_mlp_motion_audit import audit_motion


def replace_node_fields(text, node, changes):
    pattern = rf"(?ms)^{re.escape(node)}:\n.*?(?=^[a-zA-Z_]\w*:\n|\Z)"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Missing deployment node: {node}")
    block = match.group()
    for key, value in changes.items():
        line = rf"(?m)^    {re.escape(key)}:.*$"
        block, count = re.subn(line, f"    {key}: {value}", block)
        if count != 1:
            raise ValueError(f"Expected one {node}.{key} field, found {count}")
    return text[:match.start()]+block+text[match.end():]


def validate_joint_bindings(text):
    """Semantic names alias physical sources; check the robot's bindings too."""
    bridge = yaml.safe_load(text)["rl_bridge"]["ros__parameters"]
    joints = ["right_front_joint","left_front_joint","left_back_joint","right_back_joint"]
    if (bridge["joint_names"] != joints or bridge["joint_base_path"] != "/chassis"
            or bridge["joint_angle_suffix"] != "/urdf_angle"
            or bridge["joint_velocity_suffix"] != "/urdf_velocity"):
        raise ValueError("Joint bindings must match trained RF/LF/LB/RB URDF coordinates")
    terms = {int(fields["index"]): fields for fields in
             (dict(token.split("=",1) for token in term.split()) for term in bridge["observation_terms"])}
    for index in (10,14):
        if (terms[index]["joints"].split(",") != joints
                or terms[index].get("relative","false").lower() in ("true","1")
                or terms[index].get("zero","false").lower() in ("true","1")):
            raise ValueError("Joint observation bindings differ from trained RF/LF/LB/RB absolute order")
    for offset,joint in enumerate(joints):
        if terms[18+offset]["path"] != f"/chassis/{joint}/feedback_current_raw":
            raise ValueError("Current bindings differ from trained RF/LF/LB/RB order")
        wheel = joint.replace("_joint","_wheel")
        if terms[26+offset]["path"] != f"/chassis/{wheel}/rl_velocity":
            raise ValueError("Wheel encoder bindings differ from trained RF/LF/LB/RB order")


def run_logged(command, path, expected_success=True):
    with path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if (result.returncode == 0) != expected_success:
        raise RuntimeError(f"Unexpected deployment check result ({result.returncode}): {path}")
    if not expected_success and "== SUMMARY: FAIL ==" not in path.read_text(errors="replace"):
        raise RuntimeError(f"Negative check failed outside contract validation: {path}")
    return result.returncode


def audit_training(state, training_dir, baseline):
    """Recheck every segment, rather than trusting its saved success flag."""
    if (state["status"] != "completed" or state["requested_long_updates"] < 10000
            or state["total_training_updates"] < 10100 or not state["stages"]
            or not all(stage["qualified"] for stage in state["stages"])):
        raise ValueError("Deployment preparation requires completed, fully qualified 10000-update long training")
    stages = state["stages"]
    if (stages[0].get("name") != "preflight"
            or any(type(stage.get("updates")) is not int or stage["updates"] <= 0 for stage in stages)
            or [stage.get("name") for stage in stages[1:]] != [f"long_{i:02d}" for i in range(1,len(stages))]
            or sum(stage["updates"] for stage in stages) != state["total_training_updates"]
            or sum(stage["updates"] for stage in stages[1:]) != state["requested_long_updates"]
            or stages[-1]["checkpoint"] != state["last_accepted_checkpoint"]):
        raise ValueError("Deployment requires a complete, consistent sequence of actual training segments")
    outcome = integrity = None
    for stage in stages:
        checkpoint = Path(stage["checkpoint"])
        jobs=[job for job in state.get("jobs",[]) if job.get("name")=="train_"+stage["name"]]
        if len(jobs)!=1 or jobs[0].get("status")!="completed":
            raise ValueError("Deployment requires the completed training job for each segment")
        command=jobs[0].get("command",[])
        if "--checkpoint" not in command or "--max_iterations" not in command:
            raise ValueError("Deployment requires actual fine-tune command provenance")
        source=Path(command[command.index("--checkpoint")+1])
        if int(command[command.index("--max_iterations")+1])!=stage["updates"]:
            raise ValueError("Recorded training budget differs from claimed segment updates")
        update=actor_update_info(source,checkpoint)
        if not update["policy_updated"]:
            raise ValueError("Deployment cannot count unchanged actors as policy optimization")
        outcome = qualify(checkpoint,training_dir/(stage["name"]+"_evaluation"),baseline)
        if not outcome["transformer_level_recovered"]:
            raise ValueError("Training segment no longer matches qualified reports: "+stage["name"])
        integrity = checkpoint_info(checkpoint)
        if integrity["iteration"] != stage["updates"]-1:
            raise ValueError("Training segment did not finish its claimed PPO updates: "+stage["name"])
    return checkpoint, outcome, integrity


def prepare(training_dir, output_dir, rmcs_root, runtime_samples=None):
    state = json.loads((training_dir/"status.json").read_text())
    baseline = ROOT/"outputs/deformable_long2499_joint_reward_review_20261008/evaluation"
    checkpoint, assessment, integrity = audit_training(state,training_dir,baseline)
    motion = audit_motion(training_dir/(state["stages"][-1]["name"]+"_evaluation"))
    if output_dir.is_dir():
        (output_dir/"motion_assessment.json").write_text(json.dumps(motion,indent=2)+"\n")
    if not motion["motion_ready"]:
        raise ValueError("Deployment motion gate failed: baseline contact/tilt recovery does not prove parking or tracking")
    tool_dir = rmcs_root/"rmcs_ws/src/rmcs_rl/tool"
    source_config = rmcs_root/"rmcs_ws/src/rmcs_bringup/config/deformable-infantry-omni-rl.yaml"
    original = source_config.read_text()
    validate_joint_bindings(original)
    old_model = rmcs_root/"rmcs_ws/src/rmcs_rl/models/deformable_v3_transformer.onnx"
    run_logged([sys.executable,"-B",str(tool_dir/"check_policy_contract.py"),str(old_model),
                "--config",str(source_config),"--node","rl_bridge"],output_dir/"original_contract.log")
    text = replace_node_fields(original,"rl_bridge",dict(rl_obs_size=160,history_length=5))
    text = replace_node_fields(text,"policy_server",dict(
        rl_model_path='"models/deformable_mlp5_routed_v3.onnx"',model_type='"mlp"',
        sequence_length=5,feature_size=32,rl_obs_size=160,normalization_from_metadata="false"))
    validate_joint_bindings(text)
    config = output_dir/"deformable-infantry-omni-rl.yaml"
    config.write_text(text)
    relative = source_config.relative_to(rmcs_root).as_posix()
    patch = "".join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
                                         fromfile="a/"+relative,tofile="b/"+relative))
    (output_dir/"deformable-mlp5-config.patch").write_text(patch)
    run_logged([sys.executable,"-B",str(ROOT/"scripts/rsl_rl/export_onnx.py"),
                "--checkpoint",str(checkpoint),"--output_dir",str(output_dir),
                "--filename","deformable_mlp5_routed_v3.onnx"],output_dir/"export.log")
    model = output_dir/"deformable_mlp5_routed_v3.onnx"
    run_logged([sys.executable,"-B",str(tool_dir/"stamp_layout_metadata.py"),"--model",str(model),
                "--from-config",str(config),"--node","rl_bridge","--model-type","mlp",
                "--sequence-length","5","--feature-size","32","--action-clip","1",
                "--policy-version","deformable-mlp5-"+integrity["sha256"][:12]],output_dir/"stamp.log")
    spec = importlib.util.spec_from_file_location("deployment_native_layout",tool_dir/"rl_layout.py")
    layout = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = layout
    spec.loader.exec_module(layout)
    model_id = layout.model_id_hex(model)
    command = [sys.executable,"-B",str(tool_dir/"check_policy_contract.py"),str(model),
               "--config",str(config),"--node","rl_bridge","--model-type","mlp",
               "--sequence-length","5","--feature-size","32","--expect-model-id",model_id]
    run_logged(command,output_dir/"contract_positive.log")
    negative_dir = output_dir/"negative_checks"
    negative_dir.mkdir()
    parsed = yaml.safe_load(text)
    bad = copy.deepcopy(parsed)
    bad["rl_bridge"]["ros__parameters"]["history_length"] = 8
    bad["rl_bridge"]["ros__parameters"]["rl_obs_size"] = 256
    bad["policy_server"]["ros__parameters"]["sequence_length"] = 8
    bad["policy_server"]["ros__parameters"]["rl_obs_size"] = 256
    wrong_history = negative_dir/"wrong_history.yaml"
    wrong_history.write_text(yaml.safe_dump(bad,sort_keys=False))
    bad = copy.deepcopy(parsed)
    terms = bad["rl_bridge"]["ros__parameters"]["observation_terms"]
    index = next(i for i,term in enumerate(terms) if term.startswith("index=10 "))
    expected = "joints=right_front_joint,left_front_joint,left_back_joint,right_back_joint"
    if expected not in terms[index]:
        raise ValueError("Deployment joint order differs from the trained RF/LF/LB/RB contract")
    terms[index] = terms[index].replace(
        "joints=right_front_joint,left_front_joint,left_back_joint,right_back_joint",
        "joints=left_front_joint,right_front_joint,left_back_joint,right_back_joint")
    binding_only = yaml.safe_dump(bad,sort_keys=False)
    try:
        validate_joint_bindings(binding_only)
    except ValueError:
        (negative_dir/"wrong_binding_check.txt").write_text("REJECTED: trained joint binding order mismatch\n")
    else:
        raise RuntimeError("Wrong physical binding order was not rejected")
    terms[index] = terms[index].replace("name=leg_urdf_q_rf_lf_lb_rb","name=leg_urdf_q_lf_rf_lb_rb")
    wrong_layout = negative_dir/"wrong_joint_order.yaml"
    wrong_layout.write_text(yaml.safe_dump(bad,sort_keys=False))
    for path in (wrong_history,wrong_layout):
        negative = command.copy()
        negative[negative.index("--config")+1] = str(path)
        run_logged(negative,negative_dir/(path.stem+".log"),expected_success=False)
    native = None
    if runtime_samples is not None:
        from deformable_mlp_runtime_check import check
        native = check(checkpoint,model,runtime_samples,output_dir/"native_runtime",rmcs_root)
        if not native["inference_p99_below_10ms"]:
            raise ValueError("Native inference exceeds the 100 Hz policy budget on this host")
        shutil.copyfile(runtime_samples,output_dir/"runtime_samples.npz")
        shutil.copyfile(runtime_samples.with_suffix(".json"),output_dir/"runtime_samples.json")
    shutil.copyfile(checkpoint,output_dir/checkpoint.name)
    shutil.copytree(checkpoint.parent/"params",output_dir/"training_params")
    (output_dir/"final_assessment.json").write_text(json.dumps(assessment,indent=2)+"\n")
    (output_dir/"long_training_status.json").write_text(json.dumps(state,indent=2)+"\n")
    manifest = dict(status="prepared",checkpoint=str(checkpoint),checkpoint_sha256=integrity["sha256"],
                    model_id=model_id,onnx_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    input_shape=[1,160],output_shape=[1,4],history_length=5,frame_size=32,
                    normalization="folded into model; no runtime normalization",policy_rate_hz=100,
                    source_config=str(source_config),source_config_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    contracts="positive passed; wrong history/semantic joint order rejected; physical bindings checked separately",
                    hardware_validation="not performed",strict_horizontal_goal_achieved=assessment["strict_horizontal_goal_achieved"],
                    motion_readiness=motion)
    bridge=yaml.safe_load(text)["rl_bridge"]["ros__parameters"]
    obs_signature=layout.obs_signature(bridge["observation_terms"],4,history_length=5)
    action_signature=layout.action_signature(bridge["action_terms"])
    manifest["layout_hash"]=layout.hex16(layout.layout_hash(obs_signature,action_signature,160,4))
    if native:
        if (native["native_metrics"]["model_id"] != model_id
                or native["native_metrics"]["layout_hash"] != manifest["layout_hash"]):
            raise ValueError("Actual C++ metadata identity differs from final deployment metadata")
        manifest["native_runtime_validation"] = native
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (output_dir/"README.md").write_text(
        "# 五帧 MLP 部署准备\n\n"
        "该目录是完成一万轮、逐段 Isaac Sim 验收后的最终模型，尚未运行实车。\n\n"
        "配置补丁只修改 RL 节点，保留 RMCS 其他并行工作。完整 YAML 用于校验和审阅，"
        "不要用它覆盖包含后续改动的配置。模型文件目标位置与原 Transformer 一样："
        "`rmcs_ws/src/rmcs_rl/models/`。修改安装位置时保持模型字节、5 帧历史和 100 Hz 一致。\n\n"
        "应用前核对源配置哈希、最终 manifest 的 model_id、layout_hash 和原生正负检查日志。"
        "当前训练物理模型的轴力矩标定及实车运行仍需现场验证。17/20 度仍按最佳努力残余倾角验收，"
        "不能宣称全坡度三度以内调平。\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--rmcs-root",type=Path,default=Path("/home/noir/Documents/workspace/RMCS"))
    parser.add_argument("--wait",action="store_true")
    parser.add_argument("--runtime-samples",type=Path,default=None)
    args = parser.parse_args()
    args.training_dir,args.output_dir,args.rmcs_root=(path.resolve() for path in (args.training_dir,args.output_dir,args.rmcs_root))
    args.output_dir.mkdir(parents=True,exist_ok=False)
    status_path = args.output_dir/"preparation_status.json"
    try:
        last_job = None
        while True:
            state = json.loads((args.training_dir/"status.json").read_text())
            if state["status"] == "completed":
                break
            if state["status"] in ("failed","rejected") or not args.wait:
                raise ValueError("Long training has not completed successfully: "+state["status"])
            if state.get("current_job") != last_job:
                last_job = state.get("current_job")
                waiting = dict(status="waiting_for_qualified_long_training",current_job=last_job,
                               training_dir=str(args.training_dir))
                status_path.write_text(json.dumps(waiting,indent=2)+"\n");print(json.dumps(waiting),flush=True)
            time.sleep(30)
        result = prepare(args.training_dir,args.output_dir,args.rmcs_root,
                         args.runtime_samples.resolve() if args.runtime_samples else None)
        status_path.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)
    except BaseException as error:
        status_path.write_text(json.dumps(dict(status="failed",error=str(error)),indent=2)+"\n")
        raise


if __name__ == "__main__":
    main()
