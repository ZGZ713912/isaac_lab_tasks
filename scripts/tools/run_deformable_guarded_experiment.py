"""Probe startup safety, then run short acceptance and gated long PPO.

Each invocation creates a new attempt without overwriting previous evidence.
Only the registered deformable evaluation/training commands are launched.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from deformable_guarded_assess import flat_assessment, report_in
from deformable_guarded_train import EXTRA_SOURCES, SOURCES, TASK, evaluate
from deformable_guarded_reuse import evaluation_sources_match, reuse_short_run
from deformable_precision_pipeline import ROOT, run_child, timestamp, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, default=101)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.checkpoint, args.probe_checkpoint = args.checkpoint.resolve(), args.probe_checkpoint.resolve()
    args.output_base = args.output_base.resolve()
    args.output_base.mkdir(parents=True, exist_ok=True)
    number = 1
    while (args.output_base / f"attempt_{number:02d}").exists():
        number += 1
    previous = args.output_base / f"attempt_{number-1:02d}" if number > 1 else None
    args.output_dir = args.output_base / f"attempt_{number:02d}"
    args.output_dir.mkdir()
    args.timeout_hours = 96.
    names = set(SOURCES + EXTRA_SOURCES + ["scripts/tools/run_deformable_guarded_experiment.py"])
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    state = dict(status="probing", task=TASK, started_at=timestamp(), current_job=None,
                 jobs=[], sources=sources, checkpoint=str(args.checkpoint),
                 probe_checkpoint=str(args.probe_checkpoint), hardware_run=False)
    for name in sources:
        path = args.output_dir / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    write_json(args.output_dir / "status.json", state)
    write_json(args.output_base / "latest.json", dict(attempt=str(args.output_dir)))
    try:
        probes = []
        cache_ok = False
        if previous and (previous / "startup_assessment.json").is_file():
            old = json.loads((previous / "status.json").read_text())
            prior = json.loads((previous / "startup_assessment.json").read_text())
            cache_ok = (prior.get("complete") and prior.get("passed")
                        and old.get("probe_checkpoint") == str(args.probe_checkpoint)
                        and all(r["checkpoint_sha256"] == hashlib.sha256(args.probe_checkpoint.read_bytes()).hexdigest()
                                for r in prior["reports"])
                        and evaluation_sources_match(previous, old["sources"], sources))
        for grade in (2., 2.9):
            if cache_ok:
                directory = previous / f"startup_probe_{grade:g}"
                report = report_in(directory)
                if (report["seed"] != 4183 or report["steps"] != 1200
                        or report["terrain"]["grade_deg"] != grade):
                    raise ValueError("Cached startup probe differs from this experiment")
                assessment = flat_assessment(directory)
                state["startup_reused_from"] = str(previous)
            else:
                directory = evaluate(args, state, args.probe_checkpoint, f"startup_probe_{grade:g}",
                                     grade, 4183, transfer=True)
                assessment = flat_assessment(directory)
            probes.append(assessment)
            write_json(args.output_dir / "startup_assessment.json", dict(reports=probes,
                       complete=len(probes) == 2, passed=all(r["passed"] for r in probes)))
            if not assessment["passed"]:
                state.update(status="needs_iteration", reason=f"Startup probe failed at {grade:g} degrees")
                return 3
        current = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
        if sources != current:
            raise ValueError("Sources changed during startup probing; start a new attempt")
        state.update(status="training_pipeline", training_directory=str(args.output_dir / "training"))
        write_json(args.output_dir / "status.json", state)
        command = [str(ROOT / "run_gui.sh"), sys.executable, "-B", "-u",
                   "scripts/tools/deformable_guarded_train.py", "--checkpoint", str(args.checkpoint),
                   "--output-dir", str(args.output_dir / "training"), "--short-iterations", "100",
                   "--short-envs", "128", "--training-seed", str(args.training_seed),
                   "--iterations", "10000", "--chunk-iterations", "1000", "--num-envs", "512",
                   "--device", args.device, "--run"]
        if previous and (previous / "training" / "baseline_flat" / "report.json").is_file():
            baseline_state = json.loads((previous / "training" / "status.json").read_text())
            if (baseline_state["checkpoint"] == str(args.checkpoint)
                    and baseline_state["checkpoint_sha256"] == hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                    and evaluation_sources_match(previous / "training", baseline_state["sources"], sources)):
                command.extend(("--baseline-flat-dir", str(previous / "training" / "baseline_flat")))
                state["baseline_reused_from"] = str(previous / "training" / "baseline_flat")
        if previous and (previous / "training" / "status.json").is_file():
            try:
                reuse_short_run(args.checkpoint, previous / "training",
                    {name: value for name, value in sources.items()
                     if name != "scripts/tools/run_deformable_guarded_experiment.py"}, 100, args.training_seed)
            except ValueError:
                pass
            else:
                command.extend(("--reuse-short-run", str(previous / "training")))
                state["short_reused_from"] = str(previous / "training")
        run_child(command, "training_pipeline", args, state)
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
