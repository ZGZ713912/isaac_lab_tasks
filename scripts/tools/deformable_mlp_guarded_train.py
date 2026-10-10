"""Fresh PPO recovery with physics checks before and throughout long training.

Only a standalone five-frame MLP that restores the paired Transformer reports
can enter this pipeline. A short preflight and every long-training chunk must
pass the same gate. Failed candidates are retained and never replace the last
accepted checkpoint. Dry-run is the default.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

from deformable_mlp_repair_assess import GRADES, assess, load_reports
from deformable_mlp_train import SOURCES
from deformable_precision_pipeline import ROOT, latest_checkpoint, run_child, timestamp, write_json

TASK = "Robotics-Deformable-Suspension-Support-Leveling-Recovered-History-MLP-Real2Sim-v3"
EXPERIMENT = "deformable_real2sim_recovered_mlp5_v3"
ROUTED_CLASS = "ActorCriticSuspensionRoutedMLP"
ROUTED_TASK = "Robotics-Deformable-Suspension-Support-Leveling-Recovered-History-Routed-MLP-Real2Sim-v3"
ROUTED_EXPERIMENT = "deformable_real2sim_recovered_routed_mlp5_v3"


def checkpoint_info(path):
    import torch
    import yaml
    params = path.parent / "params"
    agent = yaml.safe_load((params / "agent.yaml").read_text())
    policy = agent["policy"]
    if (policy["class_name"] not in ("ActorCriticSuspensionMLP", ROUTED_CLASS) or policy["history_length"] != 5
            or policy.get("actor_obs_normalization", False) or policy.get("critic_obs_normalization", False)
            or policy["activation"] != "elu"):
        raise ValueError("Guarded recovery requires the sensor-only five-frame ELU MLP")
    env = yaml.load((params / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    for key, value in dict(policy_history_length="5", observation_space="160", state_space="40",
                           action_contract_version="minangle_physical_v3", real2sim_enabled="true",
                           real2sim_observation_version="current_fraction_v2").items():
        if env.get(key) != value:
            raise ValueError(f"Recovery checkpoint contract mismatch: {key}")
    if hashlib.sha256((params / "real2sim_model.json").read_bytes()).hexdigest() != env["real2sim_model_sha256"]:
        raise ValueError("Recovery checkpoint physics snapshot hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    if not state or not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Recovery checkpoint has nonfinite policy tensors")
    architectures = [("critic", 40, policy["critic_hidden_dims"], 1)]
    routed = policy["class_name"] == ROUTED_CLASS
    if routed:
        experts = policy["expert_hidden_dims"]
        if len(experts) < 2:
            raise ValueError("Routed recovery requires at least two MLP experts")
        architectures.extend((f"actor.experts.{index}", 160, hidden, 4) for index, hidden in enumerate(experts))
        architectures.append(("actor.router", 160, policy["router_hidden_dims"], len(experts)))
        for key in ("routing_confidence", "routing_load_threshold"):
            expected = torch.tensor(policy[key])
            if not torch.equal(state["actor."+key], expected):
                raise ValueError("Routing gate metadata differs from checkpoint tensors")
        if not .5 < policy["routing_confidence"] < 1 or policy["routing_load_threshold"] < 0:
            raise ValueError("Invalid routing confidence/load gate")
    else:
        architectures.append(("actor", 160, policy["actor_hidden_dims"], 4))
    for kind, inputs, hidden, outputs in architectures:
        if not hidden or any(type(width) is not int or width <= 0 for width in hidden):
            raise ValueError("Recovery MLP requires positive hidden-layer dimensions")
        widths = [inputs, *hidden, outputs]
        for index, (before, after) in enumerate(zip(widths, widths[1:])):
            if (state[f"{kind}.{index*2}.weight"].shape != (after, before)
                    or state[f"{kind}.{index*2}.bias"].shape != (after,)):
                raise ValueError(f"Saved {kind} tensors do not match the recorded architecture")
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), iteration=payload["iter"],
                actor_hidden_dims=policy["actor_hidden_dims"], critic_hidden_dims=policy["critic_hidden_dims"],
                policy_class=policy["class_name"],
                routed_parameters={key: policy[key] for key in ("expert_hidden_dims", "router_hidden_dims",
                                                               "routing_confidence", "routing_load_threshold")} if routed else {})


def qualify(checkpoint, directory, baseline_directory):
    status = json.loads((directory / "status.json").read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if status["status"] != "completed" or status["checkpoint_sha256"] != digest:
        raise ValueError("Promotion requires completed reports for the exact checkpoint bytes")
    baseline, candidate = load_reports(baseline_directory), load_reports(directory)
    if set(baseline) != set(GRADES) or set(candidate) != set(GRADES):
        raise ValueError("Promotion requires complete five-grade coverage")
    if any(Path(report["checkpoint"]).resolve() != checkpoint.resolve() for report in candidate.values()):
        raise ValueError("Evaluation reports belong to a different checkpoint")
    return assess(baseline, candidate)


def actor_update_info(source, candidate):
    """Value/noise-only updates do not count as policy optimization."""
    import torch
    before=torch.load(source,map_location="cpu",weights_only=False)["model_state_dict"]
    after=torch.load(candidate,map_location="cpu",weights_only=False)["model_state_dict"]
    keys=[key for key in before if key.startswith("actor.") and key.endswith((".weight",".bias"))
          and not key.startswith("actor.router.")]
    if not keys or any(key not in after or before[key].shape != after[key].shape for key in keys):
        raise ValueError("Policy update requires matching actor parameter structures")
    if any(not torch.equal(value,after.get(key)) for key,value in before.items()
           if key.startswith("actor.router.")):
        raise ValueError("Recovery router changed during PPO despite its frozen contract")
    changed=sum(not torch.equal(before[key],after[key]) for key in keys)
    return dict(changed_tensors=changed,max_parameter_delta=max(
        float((before[key]-after[key]).abs().max()) for key in keys),policy_updated=changed>0)


def training_command(args, checkpoint, name, updates, envs, info):
    routed = info.get("policy_class") == ROUTED_CLASS
    command = [sys.executable, "-B", "-u", "scripts/rsl_rl/train.py", "--task", ROUTED_TASK if routed else TASK,
            "--checkpoint", str(checkpoint), "--run_name", name, "--num_envs", str(envs),
            "--max_iterations", str(updates), "--seed", str(args.seed), "--device", args.device,
            "--headless", "--finetune_noise_std", ".015",
            "agent.policy.actor_hidden_dims=" + json.dumps(info["actor_hidden_dims"], separators=(",", ":")),
            "agent.policy.critic_hidden_dims=" + json.dumps(info["critic_hidden_dims"], separators=(",", ":"))]
    if routed:
        command.extend("agent.policy."+key+"="+json.dumps(value, separators=(",", ":"))
                       for key, value in info["routed_parameters"].items())
    replay = getattr(args, "reference_replay", None)
    if replay:
        if not routed:
            raise ValueError("Sensor replay recovery currently requires the routed MLP task")
        command.extend(["--reference_replay", str(replay), "agent.algorithm.reference_replay_weight=10.0"])
        bound = getattr(args,"reference_replay_max_delta",0.)
        if bound:
            command.append("agent.algorithm.reference_replay_max_delta="+str(bound))
    return command


def benchmark_command(args, checkpoint, directory):
    task = ROUTED_TASK if checkpoint_info(checkpoint).get("policy_class") == ROUTED_CLASS else TASK
    return [sys.executable, "-B", "-u", "scripts/tools/deformable_real2sim_benchmark.py",
            "--task", task, "--checkpoint", str(checkpoint), "--output-dir", str(directory),
            "--grades", "0", "5", "10", "17", "20", "--steps", "600", "--seed", "1234",
            "--command-profile", "play", "--command-frame", "body", "--policy-only",
            "--device", args.device, "--run"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qualified-evaluation-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path,
                        default=ROOT / "outputs/deformable_long2499_joint_reward_review_20261008/evaluation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--chunk-iterations", type=int, default=1000)
    parser.add_argument("--preflight-iterations", type=int, default=100)
    parser.add_argument("--preflight-envs", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--reference-replay", type=Path, default=None)
    parser.add_argument("--reference-replay-max-delta", type=float, default=0.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-hours", type=float, default=96.)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if (min(args.iterations, args.chunk_iterations, args.preflight_iterations) <= 0
            or not 1 <= args.num_envs <= 1024 or not 1 <= args.preflight_envs <= 1024
            or args.seed < 0 or args.seed in (1234, 4321)
            or not math.isfinite(args.timeout_hours) or args.timeout_hours <= 0):
        parser.error("Positive budgets/timeout, valid environment counts and a separate training seed required")
    if (not math.isfinite(args.reference_replay_max_delta) or args.reference_replay_max_delta < 0
            or args.reference_replay_max_delta and not args.reference_replay):
        parser.error("A finite nonnegative replay drift bound requires sensor replay")
    for key in ("checkpoint", "qualified_evaluation_dir", "baseline_dir", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.reference_replay:
        args.reference_replay = args.reference_replay.resolve()
        if not args.reference_replay.is_file() or not args.reference_replay.with_suffix(".json").is_file():
            parser.error("Sensor replay requires its saved dataset and provenance manifest")
    info = checkpoint_info(args.checkpoint)
    admission = qualify(args.checkpoint, args.qualified_evaluation_dir, args.baseline_dir)
    if not admission["transformer_level_recovered"]:
        print("REJECTED: standalone MLP has not restored the paired Transformer baseline", flush=True)
        return 2
    stages = [("preflight", args.preflight_iterations, args.preflight_envs)]
    remaining, index = args.iterations, 0
    while remaining:
        updates = min(remaining, args.chunk_iterations)
        index += 1
        stages.append((f"long_{index:02d}", updates, args.num_envs))
        remaining -= updates
    if not args.run:
        for stage, updates, envs in stages:
            print(shlex.join(training_command(args, args.checkpoint, args.output_dir.name+"_"+stage,
                                             updates, envs, info)))
            print("REQUIRED: complete paired five-grade acceptance before the next stage")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", started_at=timestamp(), jobs=[], admission=admission,
                 last_accepted_checkpoint=str(args.checkpoint), total_training_updates=0,
                 requested_long_updates=args.iterations, stages=[], source_sha256={})
    if args.reference_replay:
        state["reference_sensor_replay"] = dict(path=str(args.reference_replay),
            sha256=hashlib.sha256(args.reference_replay.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(args.reference_replay.with_suffix(".json").read_bytes()).hexdigest(),
            max_raw_mean_delta=args.reference_replay_max_delta)
    for relative in dict.fromkeys([*SOURCES, "scripts/tools/deformable_mlp_guarded_train.py",
                                  "source/agent_rl/agent_rl/rsl_rl/modules/sensor_routed_mlp.py",
                                  "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_suspension_routed_mlp.py",
                                  "scripts/tools/deformable_mlp_replay.py"]):
        data = (ROOT / relative).read_bytes()
        destination = args.output_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        state["source_sha256"][relative] = hashlib.sha256(data).hexdigest()
    write_json(args.output_dir / "status.json", state)
    source = args.checkpoint
    try:
        for stage, updates, envs in stages:
            name = args.output_dir.name+"_"+stage
            run_child(training_command(args, source, name, updates, envs, info), "train_"+stage, args, state)
            experiment = ROUTED_EXPERIMENT if info.get("policy_class") == ROUTED_CLASS else EXPERIMENT
            candidate = latest_checkpoint(experiment, name)
            info = checkpoint_info(candidate)
            if info["iteration"] != updates-1:
                raise ValueError("Fresh recovery stage did not finish its requested PPO updates")
            update=actor_update_info(source,candidate)
            if not update["policy_updated"]:
                raise ValueError("Stage changed no actor weights; value/noise-only learning is not policy optimization")
            state["total_training_updates"] += updates
            directory = args.output_dir / (stage+"_evaluation")
            run_child(benchmark_command(args, candidate, directory), "benchmark_"+stage, args, state)
            outcome = qualify(candidate, directory, args.baseline_dir)
            write_json(args.output_dir / (stage+"_assessment.json"), outcome)
            state["stages"].append(dict(name=stage, checkpoint=str(candidate), updates=updates,
                                        qualified=outcome["transformer_level_recovered"],actor_update=update))
            if not outcome["transformer_level_recovered"]:
                state.update(status="rejected", rejected_checkpoint=str(candidate), ended_at=timestamp())
                print(f"REJECTED {stage}: {candidate}", flush=True)
                return 2
            source = candidate
            state["last_accepted_checkpoint"] = str(source)
            write_json(args.output_dir / "status.json", state)
        state.update(status="completed", ended_at=timestamp(), deployment_validation="still required")
    except BaseException as error:
        state.update(status="failed", ended_at=timestamp(), error=str(error))
        raise
    finally:
        write_json(args.output_dir / "status.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
