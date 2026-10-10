"""Audit steady motion from complete Isaac Sim traces before deployment staging.

Contact/tilt baseline recovery and motion readiness are separate requirements.
Limits here are explicit engineering targets, not shaft or hardware calibration.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from deformable_mlp_repair_assess import GRADES, SCENARIOS, load_reports

DEFAULT_LIMITS = dict(parking_speed_p95_m_s=.10, linear_error_p95_m_s=.15,
                      yaw_error_p95_rad_s=.10, transition_settle_s=.50)


def steady_samples(time_s, age_s, command, settle_s, slew_limits=None):
    """Keep settled episodes and command holds; retain terminal/failure samples."""
    if (time_s.ndim != 1 or len(time_s) < 2 or age_s.shape != command.shape[:2]
            or command.shape[0] != len(time_s) or command.shape[-1] != 3
            or not all(np.isfinite(v).all() for v in (time_s, age_s, command))
            or np.any(np.diff(time_s) <= 0) or not np.isfinite(settle_s) or settle_s < 0):
        raise ValueError("Invalid trace time, episode ages, command or settling interval")
    if slew_limits is not None and (len(slew_limits) != 2 or any(
            not np.isfinite(value) or value <= 0 for value in slew_limits)):
        raise ValueError("Command slew limits must be finite and positive")
    def ramp_time(delta):
        if slew_limits is None:
            return np.zeros(delta.shape[0])
        return np.maximum(np.linalg.norm(delta[:,:2],axis=-1)/slew_limits[0],
                          np.abs(delta[:,2])/slew_limits[1])
    first_time = time_s[0] - (time_s[1] - time_s[0])
    ready = first_time + ramp_time(command[0]) + settle_s
    mask = np.zeros_like(age_s, dtype=bool)
    for step, time in enumerate(time_s):
        if step:
            changed = np.any(np.abs(command[step] - command[step-1]) > 1.e-6, axis=-1)
            ready[changed] = time_s[step-1] + ramp_time(command[step]-command[step-1])[changed] + settle_s
            reset = age_s[step] < age_s[step-1]
            ready[reset] = time_s[step-1] + ramp_time(command[step])[reset] + settle_s
        mask[step] = (age_s[step] > settle_s) & (time > ready + 1.e-9)
    return mask


def trace_motion(path, row, report, limits, slew_limits=None):
    with np.load(path, allow_pickle=False) as trace:
        columns = [str(x) for x in trace["columns"]]
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate trace metric columns")
        metrics = trace["metrics"]
        if (metrics.shape != (report["steps"], report["num_envs"], len(columns))
                or not np.isfinite(metrics).all()
                or str(trace["command_frame"].item()) != report["command_frame"]):
            raise ValueError("Trace dimensions, values or command frame disagree with report")
        indices = {name: i for i, name in enumerate(columns)}
        frame = report["command_frame"]
        command = metrics[..., [indices[f"cmd_vx_{frame}"], indices[f"cmd_vy_{frame}"], indices["cmd_wz"]]]
        time_s = trace["time_s"]
        if not np.allclose(np.diff(time_s), report["step_dt_s"], rtol=0, atol=1.e-9):
            raise ValueError("Trace cadence differs from reported physics cadence")
        mask = steady_samples(time_s, trace["episode_age_s"], command, limits["transition_settle_s"],slew_limits)
        if not mask.any():
            raise ValueError("No settled constant-command samples; motion readiness is unproven")
        linear = metrics[..., indices["linear_speed_error_m_s"]]
        yaw = metrics[..., indices["yaw_error_rad_s"]]
        stopped = np.all(np.abs(command) < 1.e-6, axis=-1) & mask
        moving = mask & ~stopped
        def p95(value, selected):
            return float(np.quantile(value[selected], .95)) if selected.any() else None
        parking = p95(linear, stopped)
        linear_p95, yaw_p95 = p95(linear, moving), p95(yaw, mask)
        checks = dict(
            parking_speed=parking is None or parking <= limits["parking_speed_p95_m_s"],
            linear_tracking=linear_p95 is None or linear_p95 <= limits["linear_error_p95_m_s"],
            yaw_tracking=yaw_p95 <= limits["yaw_error_p95_rad_s"],
            all_contact_ge_0_98=row["failure_adjusted_all_contact_rate"] >= .98,
            no_physical_termination=row["physical_terminated_resets"] == 0,
            no_terrain_boundary_failure=row["terrain_boundary_violations"] == 0,
        )
        return dict(steady_samples=int(mask.sum()), parking_samples=int(stopped.sum()),
                    moving_samples=int(moving.sum()), parking_speed_p95_m_s=parking,
                    linear_error_p95_m_s=linear_p95, yaw_error_p95_rad_s=yaw_p95,
                    checks=checks, passed=all(checks.values()), trace=str(path.resolve()),
                    trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def audit_motion(directory, limits=None):
    limits = DEFAULT_LIMITS.copy() if limits is None else limits.copy()
    if (set(limits) != set(DEFAULT_LIMITS)
            or any(not np.isfinite(value) or value <= 0 for value in limits.values())):
        raise ValueError("Motion limits must be complete, finite and positive")
    status = json.loads((directory / "status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Motion readiness requires completed independent rollouts")
    checkpoint = Path(status["checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != status["checkpoint_sha256"]:
        raise ValueError("Motion reports no longer match the actual checkpoint bytes")
    params_path = checkpoint.parent/"params"/"env.yaml"
    env = yaml.load(params_path.read_text(),Loader=yaml.BaseLoader)
    slew_limits = tuple(float(env[key]) for key in (
        "drive_linear_acceleration_limit","drive_yaw_acceleration_limit"))
    reports = load_reports(directory)
    if set(reports) != set(GRADES):
        raise ValueError("Motion readiness requires all five grades")
    results = {}
    for grade, report in reports.items():
        if (Path(report["checkpoint"]).resolve() != checkpoint.resolve()
                or report.get("training_data_controller") or set(report["results"]["POLICY"]) != SCENARIOS):
            raise ValueError("Motion audit requires the exact standalone policy and complete commands")
        cases = {case: trace_motion(Path(row["trace"]), row, report, limits,slew_limits)
                 for case, row in report["results"]["POLICY"].items()}
        results[str(grade)] = dict(passed=all(r["passed"] for r in cases.values()), cases=cases,
                                  report_sha256=hashlib.sha256((directory / f"grade_{grade}.json").read_bytes()).hexdigest())
    return dict(status="completed", checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest,
                drive_slew_limits_m_s2_and_rad_s2=slew_limits,
                drive_slew_params_sha256=hashlib.sha256(params_path.read_bytes()).hexdigest(),
                thresholds=limits, grades=results, coverage_complete=True,
                motion_ready=all(r["passed"] for r in results.values()),
                note="Necessary steady-motion gate only; declared command slew plus settling is excluded. "
                     "Tilt acceptance remains separate and unchanged; "
                     "terminal samples are retained and physical/boundary failures fail admission.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_motion(args.evaluation_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(motion_ready=result["motion_ready"],
                         grade_pass={k:v["passed"] for k,v in result["grades"].items()})))


if __name__ == "__main__":
    main()
