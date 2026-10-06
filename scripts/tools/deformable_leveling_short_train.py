"""Prepare or run one bounded leveling-first fine-tune; dry-run by default."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
TASK = "Robotics-Deformable-Suspension-Leveling-Real2Sim-v3"
EXPERIMENT = "deformable_real2sim_leveling_v3"
PYTHON = Path("/home/noir/miniconda3/envs/isaaclab/bin/python")
WARM_START = ROOT / (
    "logs/rsl_rl/deformable_real2sim_mixed_corner_v3/"
    "2026-10-05_23-02-15_real2sim_v3_dual_all_postures_small/model_100.pt"
)


def training_command(args, suffix):
    return [str(ROOT / "run_gui.sh"), str(args.python), "-B", "-u",
            str(ROOT / "scripts/rsl_rl/train.py"), "--task", TASK,
            "--checkpoint", str(args.checkpoint), "--finetune_noise_std", "0.08",
            "--experiment_name", EXPERIMENT, "--run_name", suffix,
            "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
            "--seed", str(args.seed), "--device", args.device, "--headless"]


def write_status(output, state):
    temporary = output / "status.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "status.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=WARM_START)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--iterations", type=int, default=101)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.num_envs < 1 or args.seed < 0:
        parser.error("Iterations/environments must be positive and seed nonnegative")
    args.checkpoint = args.checkpoint.resolve()
    args.python = args.python.resolve()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    suffix = "leveling_short_" + stamp
    output = (args.output_dir or ROOT / "outputs" / ("deformable_" + suffix)).resolve()
    command = training_command(args, suffix)
    print(shlex.join(command), flush=True)
    print(f"Artifacts: {output}", flush=True)
    print("Training rewards changed; performance is pending the leveling benchmark.", flush=True)
    if not args.run:
        return 0
    if output.exists():
        parser.error("Output directory exists; preserve the previous run")
    if not args.checkpoint.is_file() or not args.python.is_file():
        parser.error("The checkpoint and IsaacLab Python must exist")
    if not all((args.checkpoint.parent / "params" / name).is_file()
               for name in ("agent.yaml", "env.yaml", "real2sim_model.json")):
        parser.error("Keep the checkpoint together with its params directory")
    output.mkdir(parents=True)
    sources = ["scripts/rsl_rl/train.py", "scripts/tools/deformable_leveling_short_train.py",
               "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_cfg.py",
               "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_env.py",
               "source/agent_tasks/agent_tasks/direct/deformable_suspension/cfg_utils.py",
               "source/agent_tasks/agent_tasks/direct/deformable_suspension/agents/rsl_rl_ppo_cfg.py",
               "source/agent_rl/agent_rl/rsl_rl/algorithms/ppo_diagnostics.py"]
    for relative in sources:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    state = dict(status="training", task=TASK, command=command, run_suffix=suffix,
                 warm_start=str(args.checkpoint),
                 warm_start_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                 started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                 expected_transitions=args.iterations * args.num_envs * 24,
                 source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in sources},
                 achieved_leveling=False)
    write_status(output, state)
    try:
        result = subprocess.run(command, cwd=ROOT)
        state["training_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Training exited with code {result.returncode}")
        runs = list((ROOT / "logs/rsl_rl" / EXPERIMENT).glob("*_" + suffix))
        if len(runs) != 1:
            raise RuntimeError(f"Expected this exact run suffix once, got {len(runs)}")
        checkpoint = runs[0] / f"model_{args.iterations - 1}.pt"
        if not checkpoint.is_file():
            raise RuntimeError("Training did not produce the requested final checkpoint")
        state.update(status="training_completed_pending_leveling_evaluation", run=str(runs[0]),
                     checkpoint=str(checkpoint),
                     checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        plot = subprocess.run([str(args.python), "-B", str(ROOT / "scripts/tools/deformable_training_report.py"),
                               "--run", "LevelingShort=" + str(runs[0]),
                               "--output-dir", str(output / "charts")], cwd=ROOT)
        state["plot_exit_code"] = plot.returncode
        if plot.returncode:
            raise RuntimeError("Training completed, but exporting training curves failed")
        play = [str(ROOT / "run_gui.sh"), str(args.python), str(ROOT / "scripts/rsl_rl/play_deformable.py"),
                "--task", "Robotics-Deformable-Suspension-Rough-Keyboard-Play-Real2Sim-v3",
                "--checkpoint", str(checkpoint), "--grade-deg", "20", "--device", args.device,
                "--vx_max", "0.8", "--vy_max", "0.5", "--wz_max", "1.5",
                "--fixed_camera", "--debug_motion"]
        evaluate = [str(args.python), "-B", str(ROOT / "scripts/tools/deformable_real2sim_benchmark.py"),
                    "--checkpoint", str(checkpoint), "--output-dir", str(output / "evaluation"),
                    "--command-profile", "play", "--command-frame", "body", "--run"]
        check = [str(args.python), "-B", str(ROOT / "scripts/tools/deformable_leveling_check.py"),
                 "--candidate-dir", str(output / "evaluation"),
                 "--output", str(output / "leveling_acceptance.json")]
        state["followup_commands"] = dict(play=play, benchmark=evaluate, leveling_check=check)
        (output / "next_commands.txt").write_text(
            "\n\n".join(f"{name}:\n{shlex.join(cmd)}" for name, cmd in state["followup_commands"].items()) + "\n")
        for name, cmd in state["followup_commands"].items():
            print(f"{name}: {shlex.join(cmd)}", flush=True)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        state["ended_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        write_status(output, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
