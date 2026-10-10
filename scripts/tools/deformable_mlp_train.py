"""Audit the legacy MLP5 Joint continuation with a physics promotion gate.

Dry-run by default. Every GPU job is sequential; --run creates durable status,
source snapshots and per-job logs. Long training is rejected unless the short
candidate restores the paired Transformer contact/attitude/safety results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import sys

from deformable_precision_pipeline import ROOT, latest_checkpoint, run_child, timestamp, write_json
from deformable_mlp_repair_assess import assess, load_reports

TASK = "Robotics-Deformable-Suspension-Support-Leveling-Joint-History-MLP-Real2Sim-v3"
EXPERIMENT = "deformable_real2sim_support_leveling_mlp5_v3"
SOURCES = [
    "scripts/tools/deformable_mlp_train.py", "scripts/rsl_rl/train.py",
    "scripts/tools/deformable_mlp_repair_assess.py",
    "scripts/utils/deformable_checkpoint.py", "scripts/tools/deformable_suspension_eval.py",
    "scripts/tools/deformable_real2sim_benchmark.py", "scripts/tools/deformable_precision_pipeline.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/__init__.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_cfg.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_env.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/cfg_utils.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/agents/rsl_rl_ppo_cfg.py",
    "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_suspension_mlp.py",
    "source/agent_rl/agent_rl/rsl_rl/algorithms/ppo_diagnostics.py",
]


def training_command(args, checkpoint, suffix, updates, envs):
    return [sys.executable, "-B", "-u", "scripts/rsl_rl/train.py", "--task", TASK,
            "--checkpoint", str(checkpoint), "--resume_training", "--run_name", suffix,
            "--num_envs", str(envs), "--max_iterations", str(updates),
            "--seed", str(args.seed), "--device", args.device, "--headless"]


def benchmark_command(args, checkpoint, output):
    return [sys.executable, "-B", "-u", "scripts/tools/deformable_real2sim_benchmark.py",
            "--task", TASK, "--checkpoint", str(checkpoint), "--output-dir", str(output),
            "--grades", "0", "5", "10", "17", "20", "--steps", "600", "--seed", "1234",
            "--command-profile", "play", "--command-frame", "body", "--policy-only",
            "--timeout-hours", "2", "--device", args.device, "--run"]


def checkpoint_info(path):
    """Check the actual saved architecture, Joint reward and optimizer payload."""
    import torch
    import yaml

    params = path.parent / "params"
    agent = yaml.safe_load((params / "agent.yaml").read_text())
    env = yaml.load((params / "env.yaml").read_text(), Loader=yaml.BaseLoader)
    policy = agent["policy"]
    if (policy["class_name"] != "ActorCriticSuspensionMLP" or policy["history_length"] != 5
            or policy["actor_hidden_dims"] != [256, 128, 64]
            or policy["critic_hidden_dims"] != [256, 128, 64]):
        raise ValueError("This continuation requires the matching five-frame MLP checkpoint")
    expected = dict(policy_history_length="5", observation_space="160", state_space="40",
                    action_contract_version="minangle_physical_v3",
                    real2sim_observation_version="current_fraction_v2", real2sim_enabled="true",
                    joint_supported_leveling_weight="80.0", best_effort_tilt_weight="0.0",
                    support_gap_weight="0.0", support_load_weight="0.0", clearance_margin_weight="0.0")
    for key, value in expected.items():
        if env.get(key) != value:
            raise ValueError(f"Saved MLP Joint contract mismatch: {key}")
    for key in ("all_wheel_contact", "wheel_load_balance", "tilt_quadratic",
                "flat_orientation_x_exp", "flat_orientation_y_exp"):
        if float(env["rewards"][key]) != 0:
            raise ValueError(f"Duplicate Joint reward remains active: {key}")
    model = params / "real2sim_model.json"
    if hashlib.sha256(model.read_bytes()).hexdigest() != env["real2sim_model_sha256"]:
        raise ValueError("Saved real2sim model hash differs from its environment")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "optimizer_state_dict" not in payload:
        raise ValueError("Continuation requires saved optimizer state")
    tensors = []

    def collect(value):
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(payload)
    if not tensors or not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("Checkpoint has nonfinite tensors")
    iteration = payload["iter"]
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Checkpoint requires a nonnegative saved iteration")
    return dict(iteration=iteration, finite_tensors=len(tensors),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def summarize_benchmark(directory):
    grades = {}
    for grade in (0, 5, 10, 17, 20):
        report = json.loads((directory / f"grade_{grade}.json").read_text())
        rows = list(report["results"]["POLICY"].values())
        if len(rows) != 6 or report["history"] != 5:
            raise ValueError("Incomplete five-frame MLP benchmark")
        grades[str(grade)] = dict(
            strict_passed=report["passed"],
            worst_contact_rate=min(row["failure_adjusted_all_contact_rate"] or 0. for row in rows),
            worst_contact_and_horizontal_rate=min(row["failure_adjusted_contact_and_horizontal_rate"] or 0. for row in rows),
            worst_tilt_p95_deg=max(row["tilt_deg"]["abs_p95"] if row["tilt_deg"]["abs_p95"] is not None else 180. for row in rows),
            physical_terminations=sum(row.get("physical_terminated_resets", row["terminated_resets"]) for row in rows),
            terrain_boundary_violations=sum(row.get("terrain_boundary_violations", 0) for row in rows),
            total_terminated_resets=sum(row["terminated_resets"] for row in rows),
        )
    result = dict(grades=grades, strict_horizontal_goal_achieved=all(g["strict_passed"] for g in grades.values()),
                  note="Strict <3deg status; large-grade failure is retained as best-effort diagnostics, not waived.")
    write_json(directory.parent / f"{directory.name}_assessment.json", result)
    return result


def promotion_assessment(baseline_directory, candidate_directory):
    """All five completed, matched rollouts are required before a long run."""
    baseline = load_reports(baseline_directory)
    if set(baseline) != {0, 5, 10, 17, 20}:
        raise ValueError("Long training requires a complete five-grade Transformer baseline")
    return assess(baseline, load_reports(candidate_directory))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-iterations", type=int, default=500)
    parser.add_argument("--short-envs", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-dir", type=Path,
                        default=ROOT / "outputs/deformable_long2499_joint_reward_review_20261008/evaluation",
                        help="Completed paired Transformer reports required for promotion to long training")
    parser.add_argument("--timeout-hours", type=float, default=96.)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if (args.short_iterations <= 0 or args.iterations <= 0
            or not 1 <= args.short_envs <= 1024 or not 1 <= args.num_envs <= 1024 or args.seed < 0
            or not math.isfinite(args.timeout_hours) or args.timeout_hours <= 0):
        parser.error("Positive budgets/timeout, 1..1024 environments and nonnegative seed required")
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.baseline_dir = args.baseline_dir.resolve()
    suffix = args.output_dir.name
    if not args.run:
        print(shlex.join(training_command(args, args.checkpoint, suffix + "_joint500", args.short_iterations, args.short_envs)))
        short = Path("<short-MLP-checkpoint>")
        print(shlex.join(benchmark_command(args, short, args.output_dir / "short_evaluation")))
        print(shlex.join(training_command(args, short, suffix + "_long", args.iterations, args.num_envs)))
        print(shlex.join(benchmark_command(args, Path("<long-MLP-checkpoint>"), args.output_dir / "long_evaluation")))
        return 0
    if args.output_dir.exists():
        parser.error("Use a new output directory; previous experiments are never overwritten")
    info = checkpoint_info(args.checkpoint)
    args.output_dir.mkdir(parents=True)
    state = dict(status="running", pid=os.getpid(), started_at=timestamp(), jobs=[],
                 task=TASK, source_checkpoint=str(args.checkpoint), source_checkpoint_info=info,
                 arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                 strict_horizontal_goal_achieved=None)
    state["source_sha256"] = {}
    for relative in SOURCES:
        data = (ROOT / relative).read_bytes()
        target = args.output_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        state["source_sha256"][relative] = hashlib.sha256(data).hexdigest()
    write_json(args.output_dir / "status.json", state)
    source = args.checkpoint
    try:
        for stage, updates, envs in (("joint500", args.short_iterations, args.short_envs),
                                     ("long", args.iterations, args.num_envs)):
            expected = info["iteration"] + updates
            run_suffix = suffix + "_" + stage
            run_child(training_command(args, source, run_suffix, updates, envs), f"train_{stage}", args, state)
            source = latest_checkpoint(EXPERIMENT, run_suffix)
            info = checkpoint_info(source)
            if info["iteration"] != expected:
                raise ValueError(f"Expected final iteration {expected}, got {info['iteration']}")
            state[stage + "_checkpoint"] = str(source)
            state[stage + "_checkpoint_info"] = info
            metrics = [sys.executable, "-B", "scripts/tools/deformable_training_report.py", "--run",
                       stage + "=" + str(source.parent), "--output-dir", str(args.output_dir / f"{stage}_metrics"),
                       "--metrics-only"]
            run_child(metrics, f"metrics_{stage}", args, state)
            directory = args.output_dir / f"{stage}_evaluation"
            run_child(benchmark_command(args, source, directory), f"benchmark_{stage}", args, state)
            assessment = summarize_benchmark(directory)
            state[stage + "_assessment"] = assessment
            promotion = promotion_assessment(args.baseline_dir, directory)
            write_json(directory.parent / f"{stage}_transformer_comparison.json", promotion)
            state[stage + "_transformer_comparison"] = promotion
            write_json(args.output_dir / "status.json", state)
            if not promotion["transformer_level_recovered"]:
                state.update(status="rejected", current_job=None, checkpoint=str(source),
                             rejection_reason="Independent physics reports did not restore Transformer performance",
                             strict_horizontal_goal_achieved=assessment["strict_horizontal_goal_achieved"],
                             ended_at=timestamp())
                print(f"Rejected {stage}; no subsequent training was launched. See {directory}.")
                return 2
        state.update(status="completed", current_job=None, checkpoint=str(source), ended_at=timestamp(),
                     strict_horizontal_goal_achieved=assessment["strict_horizontal_goal_achieved"])
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", ended_at=timestamp())
        raise
    finally:
        write_json(args.output_dir / "status.json", state)
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Pipeline interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
