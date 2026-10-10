"""Compare actual Isaac Sim reports against the pre-regression Transformer.

No reward or supervised loss can promote a policy. Contact, attitude tails,
physical terminations, boundary failures and command tracking come from completed
independent rollouts using the unchanged six-command protocol.
"""

import argparse
import json
from pathlib import Path

GRADES = (0, 5, 10, 17, 20)
SCENARIOS = {"static", "forward", "lateral", "spin_positive", "spin_negative", "dynamic"}


def load_reports(directory):
    reports = {}
    for grade in GRADES:
        path = directory / f"grade_{grade}.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        if set(report["results"]["POLICY"]) != SCENARIOS:
            raise ValueError(f"Incomplete command coverage: {path}")
        reports[grade] = report
    return reports


def metrics(report):
    rows = list(report["results"]["POLICY"].values())
    return dict(
        worst_contact_rate=min(row["failure_adjusted_all_contact_rate"] for row in rows),
        worst_contact_and_horizontal_rate=min(row["failure_adjusted_contact_and_horizontal_rate"] for row in rows),
        worst_tilt_p95_deg=max(row["tilt_deg"]["abs_p95"] for row in rows),
        physical_terminations=sum(row["physical_terminated_resets"] for row in rows),
        terrain_boundary_violations=sum(row["terrain_boundary_violations"] for row in rows),
        worst_linear_speed_error_p95_m_s=max(row["linear_speed_error_m_s"]["abs_p95"] for row in rows),
        worst_yaw_error_p95_rad_s=max(row["yaw_error_rad_s"]["abs_p95"] for row in rows),
        strict_passed=all(row["acceptance"]["passed"] for row in rows),
    )


def assess(baseline, candidate):
    grades = {}
    for grade, current in candidate.items():
        previous = baseline[grade]
        if current.get("training_data_controller") or previous.get("training_data_controller"):
            raise ValueError("Training controllers cannot establish standalone policy acceptance")
        for key in ("seed", "steps", "num_envs", "command_profile", "command_frame", "step_dt_s", "settle_s"):
            if current[key] != previous[key]:
                raise ValueError(f"Unmatched {key} at {grade} degrees")
        for key in ("mode", "observation_version", "model_sha256"):
            if current["real2sim"][key] != previous["real2sim"][key]:
                raise ValueError(f"Unmatched physics {key} at {grade} degrees")
        for name, row in current["results"]["POLICY"].items():
            old_hash = previous["results"]["POLICY"][name].get("initial_state_sha256")
            if not old_hash or old_hash != row.get("initial_state_sha256"):
                raise ValueError(f"Unmatched physical reset state at {grade} degrees, {name}")
        old, new = metrics(previous), metrics(current)
        checks = dict(contact_at_least_transformer=new["worst_contact_rate"] >= old["worst_contact_rate"],
                      tilt_p95_at_most_transformer=new["worst_tilt_p95_deg"] <= old["worst_tilt_p95_deg"],
                      no_physical_termination=new["physical_terminations"] == 0,
                      boundary_failures_not_increased=new["terrain_boundary_violations"] <= old["terrain_boundary_violations"])
        grades[str(grade)] = dict(transformer=old, candidate=new, checks=checks,
                                  recovered=all(checks.values()))
    complete = set(candidate) == set(GRADES)
    return dict(coverage_complete=complete, grades=grades,
                transformer_level_recovered=complete and all(g["recovered"] for g in grades.values()),
                strict_horizontal_goal_achieved=complete and all(g["candidate"]["strict_passed"] for g in grades.values()),
                note="Exact paired metrics; geometry limits do not waive any strict acceptance check.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess(load_reports(args.baseline_dir), load_reports(args.candidate_dir))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for grade, row in report["grades"].items():
        old, new = row["transformer"], row["candidate"]
        print(f"{grade}deg contact {old['worst_contact_rate']:.5%} -> {new['worst_contact_rate']:.5%}, "
              f"tilt P95 {old['worst_tilt_p95_deg']:.4f} -> {new['worst_tilt_p95_deg']:.4f}, "
              f"physical={new['physical_terminations']}, boundary={new['terrain_boundary_violations']}")
    print(f"Transformer level recovered: {report['transformer_level_recovered']}")


if __name__ == "__main__":
    main()
