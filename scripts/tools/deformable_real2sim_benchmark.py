"""Run bounded V3 slope benchmarks in sequence, with parallel command scenarios.

Each grade retains a full JSON, per-step traces and a live subprocess status.
Dry-run by default. Existing output directories are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import signal
import sys

from deformable_precision_pipeline import ROOT, extract_report, run_child, timestamp, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", default="Robotics-Deformable-Suspension-BestEffort-Precision-Real2Sim-v3")
    parser.add_argument("--grades", nargs="+", type=float, default=[0, 5, 10, 17, 20])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-only", action="store_true")
    parser.add_argument("--real2sim-mode", choices=("nominal", "randomized"), default="nominal")
    parser.add_argument("--command-profile", choices=("stress", "play"), default="stress")
    parser.add_argument("--command-frame", choices=("world", "body"), default="world")
    parser.add_argument("--timeout-hours", type=float, default=2.)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not 50 < args.steps <= 600:
        parser.error("Use 51..600 steps, retaining a post-settle interval inside the ramp")
    if args.seed < 0 or len(set(args.grades)) != len(args.grades) or any(not 0 <= g <= 20 for g in args.grades):
        parser.error("Seed must be nonnegative; grades must be unique and in 0..20")
    args.output_dir = args.output_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    jobs = []
    for grade in args.grades:
        name = f"grade_{grade:g}"
        command = [str(ROOT / "run_gui.sh"), sys.executable, "-B", "-u",
                   "scripts/tools/deformable_suspension_eval.py", "--task", args.task,
                   "--checkpoint", str(args.checkpoint), "--num_envs", "16", "--batch-scenarios",
                   "--steps", str(args.steps), "--seed", str(args.seed), "--device", args.device,
                   "--grade-deg", str(grade), "--real2sim-mode", args.real2sim_mode,
                   "--command-profile", args.command_profile,
                   "--command-frame", args.command_frame,
                   "--trace-dir", str(args.output_dir / "traces" / name)]
        if args.policy_only:
            command.append("--policy-only")
        jobs.append((name, command))
    if not args.run:
        for name, command in jobs:
            print(f"{name}: {shlex.join(command)}")
        return 0
    if not args.checkpoint.is_file():
        parser.error("Checkpoint must exist")
    if args.output_dir.exists():
        parser.error("Output directory already exists; inspect its status before starting a new benchmark")
    args.output_dir.mkdir(parents=True)
    state = dict(status="running", started_at=timestamp(), checkpoint=str(args.checkpoint),
                 checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                 eval_source_sha256=hashlib.sha256((ROOT / "scripts/tools/deformable_suspension_eval.py").read_bytes()).hexdigest(),
                 grades=args.grades, steps=args.steps, seed=args.seed, mode=args.real2sim_mode,
                 envs_per_scenario=16, total_envs=96, jobs=[], reports=[])
    state["command_profile"] = args.command_profile
    state["command_frame"] = args.command_frame
    state["source_sha256_by_job"] = {}
    write_json(args.output_dir / "status.json", state)
    try:
        for name, command in jobs:
            state["source_sha256_by_job"][name] = {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (ROOT / "scripts/tools/deformable_suspension_eval.py",
                             ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_env.py",
                             ROOT / "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_transformer.py",
                             ROOT / "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_suspension_mlp.py")
            }
            log = run_child(command, name, args, state)
            report = extract_report(log)
            if set(report["results"]["POLICY"]) != {
                "static", "forward", "lateral", "spin_positive", "spin_negative", "dynamic",
            }:
                raise ValueError("Completed benchmark is missing a requested command scenario")
            target = args.output_dir / f"{name}.json"
            write_json(target, report)
            state["reports"].append(str(target))
            write_json(args.output_dir / "status.json", state)
        state.update(status="completed", current_job=None, ended_at=timestamp())
    except BaseException as error:
        state.update(status="failed", error=str(error), ended_at=timestamp())
        raise
    finally:
        write_json(args.output_dir / "status.json", state)
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Benchmark interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
