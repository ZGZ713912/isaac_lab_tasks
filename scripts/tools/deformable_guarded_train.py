"""Short 200Hz PPO, measured acceptance, then gated long training.

Every checkpoint and source snapshot is retained. A failed gate stops the job
before the next training chunk. No model is written to RMCS by this script.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from deformable_guarded_assess import FLAT_LIMITS, baseline_safety, flat_assessment, convergence, paired_slopes, report_in
from deformable_guarded_reuse import reuse_short_run, training_checkpoint
from deformable_mlp_guarded_train import checkpoint_info, actor_update_info
from deformable_mlp_train import SOURCES
from deformable_precision_pipeline import ROOT, run_child, timestamp, write_json

TASK = "Robotics-Deformable-Suspension-Guarded-History-MLP-Real2Sim-v4"
EXPERIMENT = "deformable_guarded_mlp5_v4"
SCENARIOS = ["static", "forward", "lateral", "spin_positive", "spin_negative", "dynamic", "accel_brake", "reversal"]
EXTRA_SOURCES = ["scripts/tools/deformable_guarded_train.py", "scripts/tools/deformable_guarded_assess.py",
    "scripts/tools/deformable_guarded_reuse.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/runtime_guard.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/training_supervision.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/adrc.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/env.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/wheel_drive.py",
    "source/agent_tasks/agent_tasks/direct/deformable_suspension/real2sim.py",
    "source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_suspension_routed_mlp.py",
    "source/agent_rl/agent_rl/rsl_rl/modules/sensor_routed_mlp.py"]


def evaluate(args, state, checkpoint, name, grade, seed, transfer=False, randomized=False, steps=None):
    directory = args.output_dir / name
    directory.mkdir(parents=True, exist_ok=False)
    scenarios = SCENARIOS if grade == 0 else SCENARIOS[:6]
    command = [str(ROOT / "run_gui.sh"), sys.executable, "-B", "-u", "scripts/tools/deformable_suspension_eval.py",
        "--task", TASK, "--checkpoint", str(checkpoint), "--num-envs", "16", "--batch-scenarios",
        "--steps", str(steps or (2400 if grade == 0 else 1200)), "--seed", str(seed), "--device", args.device,
        "--grade-deg", str(grade), "--real2sim-mode", "randomized" if randomized else "nominal",
        "--command-profile", "play", "--command-frame", "body", "--policy-only",
        "--scenarios", *scenarios, "--trace-dir", str(directory / "traces")]
    if transfer:
        command.append("--runtime-guard-transfer")
    log = run_child(command, name.replace("/", "_"), args, state)
    shutil.copyfile(log, directory / "run.log")
    report = report_in(directory)
    write_json(directory / "report.json", report)
    return directory


def train(args, state, checkpoint, name, updates, envs, transfer=False, seed=67):
    command = [str(ROOT / "run_gui.sh"), sys.executable, "-B", "-u", "scripts/rsl_rl/train.py",
        "--task", TASK, "--checkpoint", str(checkpoint), "--run_name", name,
        "--num_envs", str(envs), "--max_iterations", str(updates), "--seed", str(seed),
        "--device", args.device, "--headless", "--finetune_noise_std", ".01"]
    if transfer:
        command.append("--runtime_guard_finetune")
    log = run_child(command, name, args, state)
    candidate = training_checkpoint(log, EXPERIMENT, name)
    info = checkpoint_info(candidate)
    update = actor_update_info(checkpoint, candidate)
    if info["iteration"] != updates-1 or not update["policy_updated"]:
        raise ValueError("Training budget incomplete or actor parameters did not change")
    record = dict(checkpoint=str(candidate), checkpoint_sha256=info["sha256"], updates=updates,
                  actor_update=update, seed=seed)
    state.setdefault("training", []).append(record)
    write_json(args.output_dir / "status.json", state)
    return candidate


def verify_source(args, state):
    current = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in state["sources"]}
    if current != state["sources"]:
        raise ValueError("Training sources changed during the job; retain this run and start a new iteration")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-flat-dir", type=Path)
    parser.add_argument("--reuse-short-run", type=Path,
                        help="Reuse completed short weights; every training/runtime source must match")
    parser.add_argument("--short-iterations", type=int, default=100)
    parser.add_argument("--short-envs", type=int, default=128)
    parser.add_argument("--training-seed", type=int, default=67)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--chunk-iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-hours", type=float, default=8.)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    args.checkpoint, args.output_dir = args.checkpoint.resolve(), args.output_dir.resolve()
    info = checkpoint_info(args.checkpoint)
    if info["policy_class"] != "ActorCriticSuspensionRoutedMLP":
        parser.error("This experiment preserves the qualified three-expert H5 MLP architecture")
    if min(args.short_iterations, args.short_envs, args.chunk_iterations, args.num_envs) <= 0 or args.iterations < 0:
        parser.error("Invalid training budget")
    if not args.run:
        print(json.dumps(dict(task=TASK, short_iterations=args.short_iterations, short_envs=args.short_envs,
                              long_iterations=args.iterations, long_envs=args.num_envs, promotion="measured gates")))
        return 0
    if args.output_dir.exists():
        parser.error("Output directory exists; retain it and choose a new iteration directory")
    args.output_dir.mkdir(parents=True)
    state = dict(status="running", started_at=timestamp(), current_job=None, jobs=[], training=[],
        checkpoint=str(args.checkpoint), checkpoint_sha256=info["sha256"], task=TASK,
        short_iterations=args.short_iterations, short_completed_updates=0,
        planned_long_iterations=args.iterations,
        long_completed_updates=0, long_qualified_updates=0, last_qualified_checkpoint=None,
        hardware_run=False, requirements=dict(minangle_deg=17., flat_enter_deg=3., policy_rate_hz=200.,
                                               history_period_s=.01, frozen_router=True,
                                               flat_return_allowed_s=FLAT_LIMITS["target_check_after_s"]))
    state["sources"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in sorted(set(SOURCES+EXTRA_SOURCES))}
    for name in state["sources"]:
        target = args.output_dir / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_json(args.output_dir / "status.json", state)
    try:
        candidate = None
        if args.reuse_short_run:
            candidate, record = reuse_short_run(args.checkpoint, args.reuse_short_run,
                state["sources"], args.short_iterations, args.training_seed)
            state["training"].append(record)
            state.update(short_reused_updates=args.short_iterations, new_short_updates=0,
                         short_training_reused_from=str(args.reuse_short_run.resolve()))
            if not args.baseline_flat_dir:
                args.baseline_flat_dir = args.reuse_short_run / "baseline_flat"
        if args.baseline_flat_dir:
            base_flat = args.baseline_flat_dir.resolve()
            state["baseline_flat_reused_from"] = str(base_flat)
            report = report_in(base_flat)
            if (Path(report["checkpoint"]).resolve() != args.checkpoint or report["seed"] != 1234
                    or report["steps"] != 2400 or set(report["results"]["POLICY"]) != set(SCENARIOS)):
                raise ValueError("Cached baseline does not match this experiment")
        else:
            base_flat = evaluate(args, state, args.checkpoint, "baseline_flat", 0, 1234, transfer=True)
        baseline_flat = flat_assessment(base_flat)
        write_json(args.output_dir / "baseline_flat_assessment.json", baseline_flat)
        baseline_gate = baseline_safety(baseline_flat)
        write_json(args.output_dir / "baseline_safety_assessment.json", baseline_gate)
        if not baseline_gate["passed"]:
            state.update(status="needs_iteration", reason="Runtime guard fails flat safety before PPO")
            return 3
        if candidate is None:
            candidate = train(args, state, args.checkpoint, args.output_dir.name+"_short", args.short_iterations,
                              args.short_envs, transfer=True, seed=args.training_seed)
            state["new_short_updates"] = args.short_iterations
        state.update(short_completed_updates=args.short_iterations, short_checkpoint=str(candidate),
                     short_checkpoint_sha256=state["training"][0]["checkpoint_sha256"])
        verify_source(args, state)
        short_flat_dir = evaluate(args, state, candidate, "short_flat", 0, 1234)
        short_flat = flat_assessment(short_flat_dir)
        direction = convergence(baseline_flat, short_flat)
        gate = dict(flat=short_flat, convergence=direction,
                    passed=short_flat["passed"] and direction["passed"])
        write_json(args.output_dir / "short_assessment.json", gate)
        if not gate["passed"]:
            state.update(status="needs_iteration", reason="Short training fails measured flat/actor convergence gate")
            return 3
        # Reject changed slope behavior before spending on independent flat
        # probes. Full coverage and every independent gate remain mandatory.
        base_slopes, short_slopes = args.output_dir / "baseline_slopes", args.output_dir / "short_slopes"
        for grade in (5, 17, 20, 0, 10):
            evaluate(args, state, args.checkpoint, f"baseline_slopes/grade_{grade}", grade, 1234, transfer=True, steps=1200)
            evaluate(args, state, candidate, f"short_slopes/grade_{grade}", grade, 1234, steps=1200)
            grade_assessment = paired_slopes(base_slopes, short_slopes, grades=(grade,))
            write_json(args.output_dir / f"short_grade_{grade}_assessment.json", grade_assessment)
            if not grade_assessment["passed"]:
                state.update(status="needs_iteration", reason=f"Short training regressed paired {grade:g}-degree behavior")
                return 3
        paired = paired_slopes(base_slopes, short_slopes)
        write_json(args.output_dir / "short_slope_assessment.json", paired)
        if not paired["passed"] or not paired["complete"]:
            state.update(status="needs_iteration", reason="Short training regressed paired contact/tilt/safety")
            return 3
        # A second seed is used only after the candidate and gates are fixed.
        admission = []
        for grade in (2., 2.9, 0):
            directory = evaluate(args, state, candidate, f"admission_grade_{grade:g}", grade, 2941)
            admission.append(flat_assessment(directory))
            write_json(args.output_dir / "independent_admission.json", dict(reports=admission,
                       complete=False, passed=False))
            if not admission[-1]["passed"]:
                state.update(status="needs_iteration", reason=f"Independent {grade:g}-degree near-flat admission failed")
                return 3
        noisy = evaluate(args, state, candidate, "admission_randomized_flat", 0, 2941, randomized=True)
        admission.append(flat_assessment(noisy))
        write_json(args.output_dir / "independent_admission.json", dict(reports=admission,
                   complete=True, passed=all(r["passed"] for r in admission)))
        if not all(r["passed"] for r in admission):
            state.update(status="needs_iteration", reason="Independent near-flat/noisy admission failed")
            return 3
        verify_source(args, state)
        state.update(short_qualified=True, last_qualified_checkpoint=str(candidate), status="long_training")
        write_json(args.output_dir / "status.json", state)
        for chunk in range((args.iterations+args.chunk_iterations-1)//args.chunk_iterations):
            verify_source(args, state)
            updates = min(args.chunk_iterations, args.iterations-state["long_completed_updates"])
            trial = train(args, state, candidate, args.output_dir.name+f"_long_{chunk+1:02d}", updates,
                          args.num_envs, seed=68+chunk)
            state["long_completed_updates"] += updates
            verify_source(args, state)
            directory = evaluate(args, state, trial, f"long_{chunk+1:02d}_flat", 0, 1234)
            flat = flat_assessment(directory)
            write_json(args.output_dir / f"long_{chunk+1:02d}_flat_assessment.json", flat)
            if not flat["passed"]:
                state.update(status="needs_iteration", reason="Long chunk regressed measured flat requirements")
                return 3
            edge_reports = []
            for grade in (2., 2.9):
                edge_dir = evaluate(args, state, trial, f"long_{chunk+1:02d}_grade_{grade:g}", grade, 2941)
                edge_reports.append(flat_assessment(edge_dir))
                write_json(args.output_dir / f"long_{chunk+1:02d}_admission.json",
                           dict(reports=edge_reports, complete=False, passed=False))
                if not edge_reports[-1]["passed"]:
                    state.update(status="needs_iteration", reason="Long chunk regressed near-flat boundary requirements")
                    return 3
            noisy_dir = evaluate(args, state, trial, f"long_{chunk+1:02d}_randomized_flat", 0, 2941, randomized=True)
            edge_reports.append(flat_assessment(noisy_dir))
            write_json(args.output_dir / f"long_{chunk+1:02d}_admission.json",
                       dict(reports=edge_reports, complete=True, passed=all(r["passed"] for r in edge_reports)))
            if not edge_reports[-1]["passed"]:
                state.update(status="needs_iteration", reason="Long chunk regressed noisy flat requirements")
                return 3
            slopes = args.output_dir / f"long_{chunk+1:02d}_slopes"
            for grade in (0, 5, 10, 17, 20):
                evaluate(args, state, trial, f"long_{chunk+1:02d}_slopes/grade_{grade}", grade, 1234, steps=1200)
                grade_assessment = paired_slopes(base_slopes, slopes, grades=(grade,))
                write_json(args.output_dir / f"long_{chunk+1:02d}_grade_{grade}_assessment.json", grade_assessment)
                if not grade_assessment["passed"]:
                    state.update(status="needs_iteration", reason=f"Long chunk regressed paired {grade:g}-degree behavior")
                    return 3
            assessment = paired_slopes(base_slopes, slopes)
            write_json(args.output_dir / f"long_{chunk+1:02d}_slope_assessment.json", assessment)
            if not assessment["passed"]:
                state.update(status="needs_iteration", reason="Long chunk regressed paired slope behavior")
                return 3
            verify_source(args, state)
            candidate = trial
            state.update(last_qualified_checkpoint=str(candidate), long_qualified_updates=state["long_completed_updates"])
            write_json(args.output_dir / "status.json", state)
        state.update(status="completed", current_job=None)
        return 0
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state.update(updated_at=timestamp(), stopped_at=timestamp())
        write_json(args.output_dir / "status.json", state)


if __name__ == "__main__":
    raise SystemExit(main())
