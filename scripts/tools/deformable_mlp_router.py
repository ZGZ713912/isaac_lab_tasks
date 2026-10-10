"""Assemble sensor-only MLP experts from regionally verified checkpoints.

The classifier checkpoint must contain recorded whole-vehicle held-out data.
All expert sources need completed paired reports on their selected grades.
Assembly does not imply closed-loop acceptance of the resulting policy.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/tools"))
from deformable_mlp_distill import validate_anchor_reference
from deformable_mlp_guarded_train import checkpoint_info, ROUTED_CLASS, ROUTED_EXPERIMENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--experts", type=Path, nargs=3, required=True)
    parser.add_argument("--expert-evaluations", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    references = [validate_anchor_reference(checkpoint, directory, grades)
                  for checkpoint, directory, grades in zip(args.experts, args.expert_evaluations,
                                                           ([0, 5, 20], [10], [17]))]
    infos = [checkpoint_info(path) for path in args.experts]
    router_payload = torch.load(args.router, map_location="cpu", weights_only=False)
    routing_info = router_payload["infos"]
    if (routing_info["input_size"] != 160 or routing_info["activation"] != "elu"
            or routing_info["seed"] in (1234, 4321)):
        raise ValueError("Router must use five raw sensor frames and a separate training seed")
    spec = importlib.util.spec_from_file_location("sensor_router_assembly",
        ROOT / "source/agent_rl/agent_rl/rsl_rl/modules/sensor_routed_mlp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actor = module.SensorRoutedMLP([info["actor_hidden_dims"] for info in infos], routing_info["hidden_dims"])
    actor.router.load_state_dict(router_payload["state_dict"], strict=True)
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in args.experts]
    for expert, payload in zip(actor.experts, payloads):
        expert.load_state_dict({key[len("actor."):]: value for key, value in payload["model_state_dict"].items()
                                if key.startswith("actor.")}, strict=True)
    physics = [(path.parent/"params/real2sim_model.json").read_bytes() for path in args.experts]
    if any(value != physics[0] for value in physics):
        raise ValueError("Experts use different physical snapshots")
    held_count, high_wrong, correct, early_correct, early_count = 0, 0, 0, 0, 0
    recalls = [[0, 0], [0, 0]]
    with torch.no_grad():
        for row in routing_info["datasets"]:
            path = Path(row["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Router holdout source bytes changed")
            with np.load(path) as data:
                raw = data["policy_obs"][:, np.arange(data["policy_obs"].shape[1])%4==3, -160:]
                steps, envs, _ = raw.shape
                labels = np.zeros((steps, envs), dtype=np.int64)
                if row["grade"] == 17:
                    labels[:] = 2
                elif row["grade"] == 10:
                    scenarios = data["env_scenarios"][np.arange(data["policy_obs"].shape[1])%4==3]
                    labels[:, scenarios == "spin_negative"] = 1
                labels[np.abs(raw[..., -14:-10]).mean(-1)<=.02] = 0
                early = np.broadcast_to(((np.arange(steps)*float(data["step_dt_s"])>.5)
                                        & (np.arange(steps)*float(data["step_dt_s"])<2.))[:,None],
                                       (steps, envs)).reshape(-1)
                x, y = torch.from_numpy(raw.reshape(-1,160)), torch.from_numpy(labels.reshape(-1))
                confidence, predicted = torch.cat([actor.router(part).softmax(-1)
                                                   for part in x.split(4096)]).max(-1)
                high = confidence >= actor.routing_confidence
                high_wrong += int((high & (predicted!=y)).sum())
                correct += int((predicted==y).sum()); held_count += len(y)
                early_correct += int((predicted[early]==y[early]).sum()); early_count += int(early.sum())
                for index in (1, 2):
                    mask = y == index
                    recalls[index-1][0] += int(((predicted==index)&high&mask).sum())
                    recalls[index-1][1] += int(mask.sum())
    if high_wrong:
        raise ValueError(f"Folded router made {high_wrong} confident holdout errors")
    params = args.output_dir/"params"
    params.mkdir()
    for name in ("env.yaml", "real2sim_model.json"):
        shutil.copyfile(args.experts[0].parent/"params"/name, params/name)
    config = yaml.safe_load((args.experts[0].parent/"params/agent.yaml").read_text())
    policy = config["policy"]
    policy.update(class_name=ROUTED_CLASS, expert_hidden_dims=[info["actor_hidden_dims"] for info in infos],
                  router_hidden_dims=routing_info["hidden_dims"], routing_confidence=.995, routing_load_threshold=.02)
    config.update(experiment_name=ROUTED_EXPERIMENT, run_name=args.output_dir.name, seed=routing_info["seed"])
    (params/"agent.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    state = {key: value.clone() for key, value in payloads[0]["model_state_dict"].items()
             if not key.startswith("actor.")}
    state.update({"actor."+key: value for key, value in actor.state_dict().items()})
    source_sha = {}
    for relative in ("scripts/tools/deformable_mlp_router.py", "scripts/rsl_rl/export_onnx.py",
                     "scripts/tools/deformable_mlp_guarded_train.py",
                     "source/agent_rl/agent_rl/rsl_rl/modules/sensor_routed_mlp.py",
                     "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_suspension_routed_mlp.py",
                     "outputs/deformable_mlp5_repair_20261009/mlp_router_probe.py"):
        value = (ROOT/relative).read_bytes()
        destination = args.output_dir/"source"/relative
        destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(value)
        source_sha[relative] = hashlib.sha256(value).hexdigest()
    info = dict(kind="five_frame_sensor_only_routed_mlp", references=references,
                router_checkpoint=str(args.router.resolve()), router_sha256=hashlib.sha256(args.router.read_bytes()).hexdigest(),
                router_training_info=routing_info, source_sha256=source_sha,
                held_out_accuracy=correct/held_count, held_out_early_accuracy=early_correct/early_count,
                held_out_confident_errors=high_wrong, held_out_expert_recall=[a/b for a,b in recalls],
                closed_loop_acceptance="not yet evaluated")
    torch.save(dict(model_state_dict=state, optimizer_state_dict={}, iter=0, infos=info), args.output_dir/"model_0.pt")
    integrity = checkpoint_info(args.output_dir/"model_0.pt")
    (args.output_dir/"assembly.json").write_text(json.dumps(dict(status="completed",**integrity,**info),indent=2)+'\n')
    print(json.dumps(dict(status="completed",**integrity,held_out_accuracy=correct/held_count,
                          held_out_confident_errors=high_wrong)),flush=True)


if __name__ == "__main__":
    main()
