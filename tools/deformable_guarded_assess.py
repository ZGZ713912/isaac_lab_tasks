"""Acceptance of measured minangle, mode stability and paired slope behavior."""

from pathlib import Path
import math

import numpy as np

from deformable_precision_pipeline import extract_report

FLAT_LIMITS = dict(target_error_max_deg=1.e-4, physical_error_p95_deg=.15,
                   physical_error_max_deg=.5, flat_hold_fraction=.99,
                   parked_diagonal_peak_to_peak_deg=.3, contact_rate=.98,
                   steady_window_s=1., target_check_after_s=2.)


def checkpoint_digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_in(directory):
    directory = Path(directory)
    for name in ("run.log", "report.json"):
        path = directory / name
        if path.is_file():
            try:
                return extract_report(path)
            except ValueError:
                pass
    raise ValueError(f"No completed evaluation report in {directory}")


def flat_assessment(directory):
    directory = Path(directory)
    report = report_in(directory)
    if report.get("runtime_guard", {}).get("policy_rate_hz") != 200.:
        raise ValueError("Flat acceptance requires the actual 200Hz guarded runtime")
    if not 0 <= report["terrain"]["grade_deg"] <= 3.:
        raise ValueError("Flat acceptance requires an independently fixed grade <=3deg")
    result = dict(checkpoint=report["checkpoint"], checkpoint_sha256=checkpoint_digest(report["checkpoint"]),
                  seed=report["seed"], grade_deg=report["terrain"]["grade_deg"], limits=FLAT_LIMITS, cases={})
    baseline_q = math.atan2(.13694, .029108)-math.radians(17.)
    for name, row in report["results"]["POLICY"].items():
        with np.load(directory / "traces" / f"policy_{name}.npz") as trace:
            age = trace["episode_age_s"]
            settled = age >= FLAT_LIMITS["target_check_after_s"]
            guard = trace["runtime_guard"]
            hold = guard[..., 0] > .5
            target_error = np.abs(np.rad2deg(trace["leg_target"]-baseline_q))
            physical_error = np.abs(np.rad2deg(trace["joint_q"]-baseline_q))
            actions = trace["raw_policy_actions"]
            raw_error = np.abs(actions.clip(-1., 1.) * np.where(actions < 0., 58., 1.))
            # The sign of the diagonal difference is meaningful only above the
            # encoder/servo noise floor. Measure the amplitude at parked ends.
            diagonal = np.rad2deg((trace["joint_q"] * np.array((1., -1., 1., -1.))).sum(-1) / 2.)
            columns = trace["columns"].tolist()
            command_ids = [i for i, c in enumerate(columns) if c.startswith("cmd_")]
            parked = np.linalg.norm(trace["metrics"][..., command_ids], axis=-1) < 1.e-6
            late = trace["time_s"] >= trace["time_s"][-1]-1.5
            parked &= settled & late[:, None]
            amplitudes = [float(np.ptp(diagonal[parked[:, i], i]))
                          for i in range(diagonal.shape[1]) if parked[:, i].sum() >= 100]
            valid = settled & hold
            has_samples = bool(settled.any())
            # The hard hold is a target contract; allow the servo its measured
            # settling transient before judging physical encoder error.
            steady = settled & (trace["time_s"][:, None] >= trace["time_s"][-1] - FLAT_LIMITS["steady_window_s"])
            has_steady = bool(steady.any())
            case = dict(settled_samples=int(settled.sum()), steady_samples=int(steady.sum()),
                flat_hold_fraction=float(hold[settled].mean()) if has_samples else 0.,
                target_error_max_deg=float(target_error[settled].max()) if has_samples else None,
                physical_error_p95_deg=float(np.quantile(physical_error[settled], .95)) if has_samples else None,
                physical_error_max_deg=float(physical_error[settled].max()) if has_samples else None,
                physical_steady_error_p95_deg=float(np.quantile(physical_error[steady], .95)) if has_steady else None,
                physical_steady_error_max_deg=float(physical_error[steady].max()) if has_steady else None,
                physical_error_all_samples_p95_deg=float(np.quantile(physical_error[settled], .95)) if has_samples else None,
                physical_error_all_samples_max_deg=float(physical_error[settled].max()) if has_samples else None,
                raw_target_error_p95_deg=float(np.quantile(raw_error[valid], .95)) if valid.any() else None,
                raw_target_error_mean_deg=float(raw_error[valid].mean()) if valid.any() else None,
                parked_diagonal_peak_to_peak_deg=max(amplitudes) if amplitudes else None,
                hold_release_count=int(((hold[:-1] & ~hold[1:]) & settled[1:]).sum()),
                all_contact_rate=row["failure_adjusted_all_contact_rate"],
                tilt_p95_deg=row["tilt_deg"]["abs_p95"], terminated_resets=row["terminated_resets"],
                physical_terminated_resets=row["physical_terminated_resets"],
                terrain_boundary_violations=row["terrain_boundary_violations"],
                mean_abs_slip_m_s=row.get("mean_abs_slip_m_s"))
            checks = dict(
                settled_samples=has_samples,
                strict_minangle_target=has_samples and case["target_error_max_deg"] <= FLAT_LIMITS["target_error_max_deg"],
                measured_minangle=has_samples and case["physical_error_p95_deg"] <= FLAT_LIMITS["physical_error_p95_deg"]
                    and case["physical_error_max_deg"] <= FLAT_LIMITS["physical_error_max_deg"],
                continuous_flat_hold=case["flat_hold_fraction"] >= FLAT_LIMITS["flat_hold_fraction"],
                no_parked_alternation=not amplitudes or max(amplitudes) <= FLAT_LIMITS["parked_diagonal_peak_to_peak_deg"],
                supported=(case["all_contact_rate"] or 0.) >= FLAT_LIMITS["contact_rate"],
                no_physical_reset=case["physical_terminated_resets"] == 0,
                no_boundary_failure=case["terrain_boundary_violations"] == 0)
            case.update(checks=checks, passed=all(checks.values()))
            result["cases"][name] = case
    result["passed"] = bool(result["cases"]) and all(row["passed"] for row in result["cases"].values())
    return result


def baseline_safety(assessment):
    """A safe but biased reference may train; a candidate must pass every gate."""
    keys = ("settled_samples", "strict_minangle_target", "continuous_flat_hold",
            "no_parked_alternation", "supported", "no_physical_reset", "no_boundary_failure")
    cases = {name: {key: row["checks"][key] for key in keys}
             for name, row in assessment["cases"].items()}
    return dict(cases=cases, passed=bool(cases) and all(all(row.values()) for row in cases.values()),
                scope="Reference safety only. Candidate encoder precision still requires full flat_assessment.")


def paired_slopes(baseline, candidate, grades=(0, 5, 10, 17, 20)):
    grades = tuple(grades)
    if not grades or len(set(grades)) != len(grades) or any(g not in (0, 5, 10, 17, 20) for g in grades):
        raise ValueError("Invalid paired grade coverage")
    result = dict(grades={}, complete=set(grades) == {0, 5, 10, 17, 20},
                  limits=dict(contact_drop=.002, tilt_increase_deg=.15))
    for grade in grades:
        old = extract_report(Path(baseline) / f"grade_{grade}" / "run.log")
        new = extract_report(Path(candidate) / f"grade_{grade}" / "run.log")
        for key in ("seed", "num_envs", "steps", "step_dt_s", "command_profile", "command_frame", "history"):
            if old[key] != new[key]:
                raise ValueError(f"Unmatched paired slope setting: {key}")
        cases = {}
        for name, row in old["results"]["POLICY"].items():
            other = new["results"]["POLICY"][name]
            if row["initial_state_sha256"] != other["initial_state_sha256"]:
                raise ValueError("Paired slope initial states differ")
            contact_delta = other["failure_adjusted_all_contact_rate"]-row["failure_adjusted_all_contact_rate"]
            tilt_delta = other["tilt_deg"]["abs_p95"]-row["tilt_deg"]["abs_p95"]
            passed = (contact_delta >= -.002 and tilt_delta <= .15
                      and other["terminated_resets"] <= row["terminated_resets"])
            cases[name] = dict(passed=passed, contact_delta=contact_delta, tilt_p95_delta_deg=tilt_delta,
                               contact=other["failure_adjusted_all_contact_rate"],
                               tilt_p95_deg=other["tilt_deg"]["abs_p95"],
                               terminated_resets=other["terminated_resets"])
        result["grades"][str(grade)] = dict(cases=cases, passed=all(c["passed"] for c in cases.values()))
    result["passed"] = all(g["passed"] for g in result["grades"].values())
    return result


def convergence(baseline, candidate):
    comparisons = {}
    for name, row in baseline["cases"].items():
        before, after = row["raw_target_error_mean_deg"], candidate["cases"][name]["raw_target_error_mean_deg"]
        if before is None or after is None:
            raise ValueError("Flat convergence requires valid raw policy samples")
        comparisons[name] = dict(before_deg=before, after_deg=after,
                                 passed=after <= .1 or after <= .7*before)
    return dict(cases=comparisons, passed=all(c["passed"] for c in comparisons.values()),
                scope="Raw actor direction only. Exact minangle is guaranteed by the runtime projection.")
