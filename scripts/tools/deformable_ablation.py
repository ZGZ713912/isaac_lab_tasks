"""Render or sequentially run the Minangle V2 model/history/seed ablation."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
TASK = "Robotics-Deformable-Suspension-Rough-History-Transformer-v2"
EXPERIMENT = "deformable_minangle_v2_ablation"
MODELS = {"mlp": "ActorCriticSuspensionMLP", "transformer": "ActorCriticTransformer"}
HISTORIES = (1, 4, 8)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Render only (default); no simulation imports")
    mode.add_argument("--run", action="store_true", help="Actually launch sequential subprocesses")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--num-envs", type=int, default=128, help="1 to 128 environments")
    p.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    p.add_argument("--evaluate", action="store_true", help="Evaluate each newly saved checkpoint")
    p.add_argument("--timeout", type=float, default=3600, help="Seconds per training/evaluation subprocess")
    return p


def train_command(args, model, history, seed, run_name):
    return [
        sys.executable, "scripts/rsl_rl/train.py", "--headless", "--task", TASK,
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
        "--seed", str(seed), "--experiment_name", EXPERIMENT, "--run_name", run_name,
        f"agent.policy.class_name={MODELS[model]}",
        f"agent.policy.history_length={history}", f"env.policy_history_length={history}",
        f"env.observation_space={32 * history}",
        "agent.policy.noise_std_type=log", "agent.policy.init_noise_std=0.3",
        "agent.policy.min_noise_std=0.03",
        "agent.policy.actor_hidden_dims=[256,128,64]",
        "agent.policy.critic_hidden_dims=[256,128,64]",
    ]


def evaluation_command(args, checkpoint, history, seed):
    return [
        sys.executable, "scripts/tools/deformable_suspension_eval.py", "--headless",
        "--task", TASK, "--checkpoint", str(checkpoint), "--history", str(history),
        "--seed", str(seed), "--num_envs", str(args.num_envs),
    ]


def latest_checkpoint(root, run_name):
    runs = sorted(root.glob(f"*_{run_name}"))
    if not runs:
        raise FileNotFoundError(f"No saved run with suffix {run_name} in {root}")
    checkpoints = [p for p in runs[-1].glob("model_*.pt") if p.stem[6:].isdigit()]
    if not checkpoints:
        raise FileNotFoundError(f"No saved checkpoint in {runs[-1]}")
    checkpoint = max(checkpoints, key=lambda p: int(p.stem[6:]))
    if not (checkpoint.parent / "params/agent.yaml").is_file():
        raise FileNotFoundError(f"Missing saved params/agent.yaml in {checkpoint.parent}")
    return checkpoint


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.iterations < 1 or not 1 <= args.num_envs <= 128:
        p.error("iterations must be positive and num-envs must be in 1..128")
    if not 0 < args.timeout < float("inf") or any(seed < 0 for seed in args.seeds):
        p.error("timeout must be positive and finite; seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        p.error("seeds and models must not contain duplicates")
    batch = uuid.uuid4().hex
    results = []
    for model in args.models:
        for history in HISTORIES:
            for seed in args.seeds:
                run_name = f"ablation_{batch}_{model}_h{history}_s{seed}"
                command = train_command(args, model, history, seed, run_name)
                print(shlex.join(command), flush=True)
                if not args.run:
                    if args.evaluate:
                        checkpoint = ROOT / "logs/rsl_rl" / EXPERIMENT / f"<timestamp>_{run_name}" / "<newest-model>.pt"
                        print(shlex.join(evaluation_command(args, checkpoint, history, seed)), flush=True)
                    continue
                started = time.monotonic()
                result = {"model": model, "history": history, "seed": seed, "run_name": run_name}
                phase = "train"
                try:
                    subprocess.run(command, cwd=ROOT, timeout=args.timeout, check=True)
                    checkpoint = latest_checkpoint(ROOT / "logs/rsl_rl" / EXPERIMENT, run_name)
                    result["checkpoint"] = str(checkpoint)
                    if args.evaluate:
                        phase = "evaluate"
                        command = evaluation_command(args, checkpoint, history, seed)
                        print(shlex.join(command), flush=True)
                        subprocess.run(command, cwd=ROOT, timeout=args.timeout, check=True)
                    result["status"] = "completed"
                except (subprocess.SubprocessError, OSError) as error:
                    result.update(status="failed", phase=phase, error=str(error))
                    print(f"FAILED {run_name} ({phase}): {error}", file=sys.stderr, flush=True)
                result["seconds"] = round(time.monotonic() - started, 3)
                results.append(result)
    if args.run:
        print(json.dumps({"jobs": results}, indent=2), flush=True)
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
