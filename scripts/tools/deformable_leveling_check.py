"""Compare matched slope reports; contact alone cannot accept a leveling run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deformable_suspension_eval import acceptance

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "outputs/deformable_real2sim_training_20261005/dual_all_postures_100_gate"
GRADES = (0, 5, 10, 17, 20)
SCENARIOS = {"static", "forward", "lateral", "spin_positive", "spin_negative", "dynamic"}


def compare_reports(candidate, baseline, min_steep_improvement=0.20):
    identity = {field: candidate[field] == baseline[field] for field in (
        "seed", "steps", "step_dt_s", "num_envs", "total_envs", "history",
        "command_profile", "command_frame", "friction", "policy_preprocessing")}
    identity.update(
        grade=candidate["terrain"]["grade_deg"] == baseline["terrain"]["grade_deg"],
        model=candidate["real2sim"]["model_sha256"] == baseline["real2sim"]["model_sha256"],
        mode=candidate["real2sim"]["mode"] == baseline["real2sim"]["mode"],
        scenarios=set(candidate["results"]["POLICY"]) == set(baseline["results"]["POLICY"]) == SCENARIOS,
    )
    if not all(identity.values()):
        raise ValueError(f"Unmatched benchmark conditions: {identity}")
    grade = candidate["terrain"]["grade_deg"]
    if grade not in GRADES:
        raise ValueError("The leveling gate requires grades 0, 5, 10, 17, 20")
    cases = {}
    for name, row in candidate["results"]["POLICY"].items():
        old = baseline["results"]["POLICY"][name]
        if row["initial_state_sha256"] != old["initial_state_sha256"]:
            raise ValueError(f"Unmatched initial state at {grade} deg / {name}")
        p95, old_p95 = row["tilt_deg"]["abs_p95"], old["tilt_deg"]["abs_p95"]
        valid = row["settled_samples"] > 0 and p95 is not None and old_p95 is not None
        safe = (row["terminated_resets"] == 0
                and row["physical_terminated_resets"] == 0
                and row["terrain_boundary_violations"] == 0
                and (row["failure_adjusted_all_contact_rate"] or 0.) >= .98)
        strict = valid and acceptance(row)["passed"]
        improved = valid and p95 <= old_p95 * (1. - min_steep_improvement)
        cases[name] = dict(
            safe_contact=safe, strict_horizontal_passed=strict,
            tilt_p95_deg=p95, baseline_tilt_p95_deg=old_p95,
            tilt_reduction_fraction=(1. - p95 / old_p95 if valid and old_p95 > 0 else None),
            accepted=safe and (strict if grade <= 10 else improved),
            linear_speed_error_rms_m_s=row.get("linear_speed_error_m_s", {}).get("rms"),
            baseline_linear_speed_error_rms_m_s=old.get("linear_speed_error_m_s", {}).get("rms"),
            mean_abs_slip_m_s=row.get("mean_abs_slip_m_s"),
            leg_target_limit_fraction=row.get("target_limit_fraction"),
            leg_current_saturation_rate=row.get("leg_torque_saturation_rate"),
        )
    return dict(grade_deg=grade, identity_checks=identity, cases=cases,
                strict_horizontal_passed=all(r["strict_horizontal_passed"] for r in cases.values()),
                leveling_stage_accepted=all(r["accepted"] for r in cases.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE)
    parser.add_argument("--min-steep-improvement", type=float, default=.20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.min_steep_improvement < 1:
        parser.error("Steep improvement must be a fraction strictly between 0 and 1")
    if args.output.exists():
        parser.error("Output exists; preserve the previous acceptance record")
    rows = []
    checkpoint = None
    for grade in GRADES:
        files = [root / f"grade_{grade}.json" for root in (args.candidate_dir, args.baseline_dir)]
        candidate, baseline = (json.loads(path.read_text()) for path in files)
        if checkpoint is None:
            checkpoint = candidate["checkpoint"]
        if candidate["checkpoint"] != checkpoint:
            raise ValueError("Candidate grades must all evaluate the same checkpoint")
        result = compare_reports(candidate, baseline, args.min_steep_improvement)
        result["reports"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for path in files]
        rows.append(result)
    report = dict(
        checkpoint=checkpoint,
        criteria={"grades_0_5_10": "Every scenario must pass the existing strict <3 deg acceptance",
                  "grades_17_20": f"Every scenario P95 must improve by at least {args.min_steep_improvement:.0%}",
                  "all_grades": "No physical/boundary resets; failure-adjusted contact >=98%"},
        leveling_stage_accepted=all(r["leveling_stage_accepted"] for r in rows),
        strict_horizontal_goal_achieved=all(r["strict_horizontal_passed"] for r in rows),
        limits="Steep improvement is a training-stage gate, not achievement of strict horizontal or vehicle validation",
        grades=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "grades"}, indent=2))
    return 0 if report["leveling_stage_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
