"""Run two fine-tunes and automatic Foundation / 10,17,20-degree evaluations.

Dry-run by default. Run sequentially with Isaac Lab's Python and --run.
Every subprocess has its own log; status.json survives desktop disconnects.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FOUNDATION = "Robotics-Deformable-Suspension-BestEffort-Foundation-v2"
PRECISION = "Robotics-Deformable-Suspension-BestEffort-Precision-v2"
SCENARIOS = ["static", "forward", "lateral", "spin_positive", "spin_negative", "dynamic"]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def extract_report(path):
    for line in reversed(path.read_text(errors="replace").splitlines()):
        if line.startswith("{"):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "results" in result and "checkpoint" in result:
                return result
    raise ValueError(f"No completed evaluation JSON in {path}")


def latest_checkpoint(experiment, suffix):
    runs = list((ROOT / "logs/rsl_rl" / experiment).glob(f"*_{suffix}"))
    if len(runs) != 1:
        raise ValueError(f"Expected one run ending {suffix}, got {len(runs)}")
    checkpoints = [p for p in runs[0].glob("model_*.pt") if p.stem[6:].isdigit()]
    if not checkpoints or not (runs[0] / "params/agent.yaml").is_file():
        raise FileNotFoundError(f"Incomplete training run {runs[0]}")
    return max(checkpoints, key=lambda p: int(p.stem[6:]))


def evaluation_command(args, checkpoint, grade=None):
    cmd = [sys.executable, "-B", "scripts/tools/deformable_suspension_eval.py",
           "--task", FOUNDATION, "--checkpoint", str(checkpoint), "--policy-only",
           "--num_envs", str(args.eval_envs), "--steps", str(args.eval_steps),
           "--seed", "1234", "--device", args.device]
    if grade is not None:
        cmd += ["--grade-deg", str(grade)]
    return cmd


def training_command(args, task, suffix):
    return [sys.executable, "scripts/rsl_rl/train.py", "--task", task,
            "--checkpoint", str(args.checkpoint), "--finetune_noise_std", "0.15",
            "--experiment_name", "deformable_foundation_precision_v2", "--run_name", suffix,
            "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
            "--seed", "42", "--device", args.device, "--headless",
            "env.motion_curriculum_iterations=1", "agent.algorithm.learning_rate=0.0001",
            "agent.algorithm.entropy_coef=0.001"]


def score(report):
    """Prefer safe contact-qualified leveling, then the worst-case tilt tail."""
    scenarios = report["results"]["POLICY"].values()
    if not scenarios:
        raise ValueError("Cannot rank an empty evaluation")
    rows = list(scenarios)
    safe = all(r["terminated_resets"] == 0 for r in rows)
    contact = all((r["failure_adjusted_all_contact_rate"] or 0) >= .98 for r in rows)
    joint = min(r["failure_adjusted_contact_and_horizontal_rate"] or 0 for r in rows)
    p95 = max(r["tilt_deg"]["abs_p95"] if r["tilt_deg"]["abs_p95"] is not None else 180. for r in rows)
    return int(safe), int(contact), joint, -p95


def run_child(command, name, args, state):
    log = args.output_dir / f"{name}.log"
    record = {"name": name, "command": command, "log": str(log), "started_at": timestamp(), "status": "running"}
    state["jobs"].append(record)
    state["current_job"] = name
    print(f"START {name}: {shlex.join(command)}", flush=True)
    task_env = os.environ.copy()
    task_env["LD_LIBRARY_PATH"] = "/home/noir/.local/lib" + (
        ":" + task_env["LD_LIBRARY_PATH"] if task_env.get("LD_LIBRARY_PATH") else "")
    with log.open("w") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=task_env, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        record["pid"] = child.pid
        deadline = time.monotonic() + args.timeout_hours * 3600
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Subprocess exceeded {args.timeout_hours} hours")
                text = log.read_text(errors="replace")
                progress = re.findall(r"Learning iteration\s+(\d+)/(\d+)", text)
                eval_progress = re.findall(r"EVAL_PROGRESS ([^\n]+)", text)
                if progress:
                    record["iteration"] = int(progress[-1][0])
                    record["iteration_budget"] = int(progress[-1][1])
                if eval_progress:
                    record["evaluation_progress"] = eval_progress[-1]
                state["updated_at"] = timestamp()
                write_json(args.output_dir / "status.json", state)
                time.sleep(15)
            if child.returncode:
                raise RuntimeError(f"{name} exited {child.returncode}; see {log}")
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            record.update(status="failed", ended_at=timestamp())
            write_json(args.output_dir / "status.json", state)
            raise
    record.update(status="completed", exit_code=child.returncode, ended_at=timestamp())
    write_json(args.output_dir / "status.json", state)
    print(f"DONE {name}", flush=True)
    return log


def write_summary(args, reports, grade_reports, winner, geometry):
    lines = ["# Deformable precision experiment", "", f"Selected checkpoint: `{winner}`", "",
             "Selection prioritizes no physical failures and >=98% contact, then worst joint horizontal success and tilt P95.",
             "A selected checkpoint is not automatically an accepted policy.", "",
             "| Candidate | Worst contact + <3deg | Worst tilt P95 | Foundation accepted |",
             "| --- | --- | --- | --- |"]
    for name, r in reports.items():
        rank = score(r)
        lines.append(f"| {name} | {rank[2]:.2%} | {-rank[3]:.3f} deg | {r['passed']} |")
    lines += ["", "## Large-grade diagnostics", "",
              "| Candidate | Grade | Scenario | Four contact | Contact + <3deg | Tilt P95 | Physical failures |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, grade, r in grade_reports:
        for scenario, row in r["results"]["POLICY"].items():
            lines.append(f"| {name} | {grade:g} deg | {scenario} | {row['all_contact_rate']} | "
                         f"{row['contact_and_horizontal_rate']} | {row['tilt_deg']['abs_p95']} | {row['terminated_resets']} |")
    lines += ["", "## Geometric reference", "",
              "The reference enforces all four sphere contacts, q in [0,Q_LOW] and 6 mm normal chassis clearance.",
              "Nonzero best feasible tilt is a multistart geometric witness, not a certified global or dynamic optimum.",
              "Torque, wheel friction and leg speed can make dynamic performance worse. Large grades are outside Foundation training."]
    for grade in args.grades:
        cases = [c for c in geometry["best_effort_cases"] if c["slope_deg"] == grade]
        values = [c["best_feasible_tilt_deg"] for c in cases if c["best_feasible_tilt_deg"] is not None]
        if values:
            lines.append(f"- {grade:g} deg: best feasible tilt {min(values):.3f}..{max(values):.3f} deg over 24 headings; "
                         f"horizontal feasible in {sum(c['horizontal_feasible'] for c in cases)}/24 headings.")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=600)
    parser.add_argument("--grades", type=float, nargs="+", default=[10., 17., 20.])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-hours", type=float, default=4.)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations <= 0 or not 1 <= args.num_envs <= 128 or not 1 <= args.eval_envs <= 24:
        parser.error("Positive iterations, 1..128 training envs and 1..24 evaluation envs required")
    if not 100 <= args.eval_steps <= 600 or not math.isfinite(args.timeout_hours) or args.timeout_hours <= 0:
        parser.error("Evaluation budget must be 100..600 steps and timeout positive")
    if not args.grades or any(not 0 <= grade <= 20 for grade in args.grades) or len(set(args.grades)) != len(args.grades):
        parser.error("Use unique grades in [0,20]")
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    trials = [("low_entropy", FOUNDATION), ("tight_tilt", PRECISION)]
    suffixes = {name: f"{args.output_dir.name}_{name}" for name, _ in trials}
    if not args.run:
        for name, task in trials:
            print(shlex.join(training_command(args, task, suffixes[name])))
        print(shlex.join(evaluation_command(args, args.checkpoint)))
        for grade in args.grades:
            print(shlex.join(evaluation_command(args, args.checkpoint, grade)))
        return 0
    if not args.checkpoint.is_file() or not (args.checkpoint.parent / "params/env.yaml").is_file():
        parser.error("Checkpoint and its saved environment config are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / "pipeline.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output_dir / "status.json").exists():
        raise ValueError("Output already has a status; use a fresh directory to preserve previous runs")
    state = {"status": "running", "pid": os.getpid(), "started_at": timestamp(), "jobs": [],
             "source_checkpoint": str(args.checkpoint), "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    state["source_checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    paths = ["scripts/rsl_rl/train.py", "scripts/rsl_rl/finetune_utils.py", "scripts/tools/deformable_suspension_eval.py",
             "scripts/tools/deformable_feasibility.py", "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_cfg.py",
             "source/agent_tasks/agent_tasks/direct/deformable_suspension/dynamic_env.py"]
    state["code_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
    write_json(args.output_dir / "status.json", state)
    reports, grade_reports = {}, []
    try:
        candidates = {"original": args.checkpoint}
        for name, task in trials:
            run_child(training_command(args, task, suffixes[name]), f"train_{name}", args, state)
            checkpoint = latest_checkpoint("deformable_foundation_precision_v2", suffixes[name])
            if int(checkpoint.stem[6:]) < args.iterations - 1:
                raise ValueError("Training ended before the requested checkpoint")
            candidates[name] = checkpoint
            state[name + "_checkpoint"] = str(checkpoint)
        for name, checkpoint in candidates.items():
            log = run_child(evaluation_command(args, checkpoint), f"foundation_{name}", args, state)
            reports[name] = extract_report(log)
            write_json(args.output_dir / f"foundation_{name}.json", reports[name])
        winner_name = max(reports, key=lambda name: score(reports[name]))
        winner = candidates[winner_name]
        state.update(selected_name=winner_name, selected_checkpoint=str(winner), foundation_accepted=reports[winner_name]["passed"])
        write_json(args.output_dir / "status.json", state)
        selected = {"original": args.checkpoint}
        if winner_name != "original":
            selected[winner_name] = winner
        # Grade tests still run when Foundation acceptance fails; they remain diagnostics.
        for name, checkpoint in selected.items():
            for grade in args.grades:
                label = f"grade_{grade:g}_{name}"
                log = run_child(evaluation_command(args, checkpoint, grade), label, args, state)
                report = extract_report(log)
                grade_reports.append((name, grade, report))
                write_json(args.output_dir / f"{label}.json", report)
        geometry_log = run_child([sys.executable, "scripts/tools/deformable_feasibility.py", "--best-effort",
                                  "--slopes", *[str(g) for g in args.grades]], "geometric_reference", args, state)
        geometry = json.loads(geometry_log.read_text())
        write_json(args.output_dir / "geometric_reference.json", geometry)
        write_summary(args, reports, grade_reports, winner, geometry)
        state.update(status="completed", current_job=None, ended_at=timestamp())
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", ended_at=timestamp())
        write_json(args.output_dir / "status.json", state)
        raise
    write_json(args.output_dir / "status.json", state)
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Pipeline interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
