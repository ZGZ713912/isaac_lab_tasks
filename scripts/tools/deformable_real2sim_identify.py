#!/usr/bin/env python3
"""Audit deformable joint sweep logs and emit a conservative real-to-sim model.

The CSVs produced by RMCS are angle-reference/closed-loop experiments.  They
contain a current-domain command and a legacy current-derived torque proxy, but
the sidecar explicitly says that the proxy is not calibrated shaft torque.  This
tool therefore keeps the proxy for diagnostics and writes a separate, explicit
simulation-effort scale.  The latter is an uncalibrated simulation prior and
must be replaced by a force-cell or joint-torque calibration when available.

Example::

    python scripts/tools/deformable_real2sim_identify.py \
      --csv /home/noir/Documents/workspace/deformable扫频数据慢速/deformable_angle_sweep_suspended_*.csv \
      --csv /home/noir/Documents/workspace/deformable扫频数据慢速/deformable_angle_sweep_grounded_*.csv \
      --csv /home/noir/Documents/workspace/deformable扫频数据慢速加高速悬空/*.csv \
      --csv /home/noir/Documents/workspace/deformable阶越/*.csv \
      --output source/agent_tasks/agent_tasks/direct/deformable_suspension/configs/deformable_real2sim.json

The command accepts shell-expanded paths and may be run without Isaac Sim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import statistics
from collections import defaultdict
from typing import Iterable


JOINTS = ("left_front", "left_back", "right_back", "right_front")
ANGLE_MIN_DEG = 17.0
ANGLE_MAX_DEG = 75.0
RAW_CURRENT_LIMIT = 2048.0
SIM_EFFORT_LIMIT_NM = 25.0
PROTOCOL_CURRENT_MAX_A = 33.0
MOTOR_TORQUE_CONSTANT_NOMINAL = 0.3
MOTOR_REDUCTION_RATIO = 36.0


def _finite(value: float | None) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


class _Stats:
    """Streaming statistics; samples are retained only for percentiles."""

    def __init__(self, sample_limit: int = 16_000) -> None:
        self.n = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.mean = 0.0
        self.m2 = 0.0
        self.samples: list[float] = []
        self.sample_limit = sample_limit
        self.rng = random.Random(0)

    def add(self, value: float | None) -> None:
        if not _finite(value):
            return
        value = float(value)
        self.n += 1
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        if len(self.samples) < self.sample_limit:
            self.samples.append(value)
        else:
            # Seeded reservoir sampling covers the entire recording without
            # favoring late sweep stages. Quantiles are approximate; mean/std
            # use every finite sample.
            index = self.rng.randrange(self.n)
            if index < self.sample_limit:
                self.samples[index] = value

    def as_dict(self) -> dict[str, float | int | None]:
        if self.n == 0:
            return {"n": 0}
        values = sorted(self.samples)

        def quantile(probability: float) -> float:
            if not values:
                return float("nan")
            position = (len(values) - 1) * probability
            lower = int(position)
            upper = min(lower + 1, len(values) - 1)
            fraction = position - lower
            return values[lower] * (1.0 - fraction) + values[upper] * fraction

        return {
            "n": self.n,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0,
            "p01": quantile(0.01),
            "p50": quantile(0.50),
            "p95": quantile(0.95),
            "p99": quantile(0.99),
        }


def _float(row: list[str], indices: dict[str, int], name: str) -> float:
    index = indices.get(name)
    if index is None or index >= len(row) or row[index] == "":
        return float("nan")
    try:
        value = float(row[index])
    except ValueError:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _joint_signal(joint: str, signal: str) -> str:
    return f"/chassis/{joint}_joint/{signal}"


def _median_or_default(values: Iterable[float], default: float) -> float:
    finite_values = [float(value) for value in values if _finite(value)]
    return statistics.median(finite_values) if finite_values else default


def _stage_stats() -> dict[str, dict[str, _Stats | int]]:
    signals = ("angle", "target", "velocity", "current", "command_current", "error", "current_saturated")
    return {
        str(stage): {
            "rows": 0,
            **{signal: _Stats() for signal in signals},
        }
        for stage in range(1, 16)
    }


def _record_period(item, kind, timestamp, sequence):
    """CSV samples skip CAN packets; estimate the interval per sequence step."""
    if not (_finite(timestamp) and _finite(sequence)):
        return
    time_key, sequence_key = f"last_{kind}_time_s", f"last_{kind}_sequence"
    previous_time, previous_sequence = item[time_key], item[sequence_key]
    if _finite(previous_time) and _finite(previous_sequence):
        if sequence == previous_sequence:
            item["sequence_duplicates"][kind] += 1
            return
        if sequence > previous_sequence and timestamp > previous_time:
            item[f"{kind}_period_s"].add((timestamp - previous_time) / (sequence - previous_sequence))
    item[time_key], item[sequence_key] = timestamp, sequence


def _audit_file(path: pathlib.Path) -> dict:
    sidecar_path = pathlib.Path(str(path) + ".json")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"missing sidecar: {sidecar_path}")
    metadata = json.loads(sidecar_path.read_text())
    if metadata.get("schema_version") != 2:
        raise ValueError(f"{path}: unsupported schema_version={metadata.get('schema_version')}")
    if metadata.get("command_domain") != "angle_reference":
        raise ValueError(f"{path}: command_domain must be angle_reference")
    if metadata.get("torque_calibrated") is not False:
        raise ValueError(f"{path}: expected an explicitly uncalibrated current-domain experiment")
    status_path = pathlib.Path(str(path) + ".status.json")
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    with path.open("rb") as binary:
        csv_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    controller_settings = list(metadata.get("experiment_config", {}).get("adrc", {}).values())
    expected = {"output_domain": "current_raw", "controller_output_to_current_raw": 5.74635241301908,
                "dt": .001, "b0": -1., "kt": 1., "td_h": .001, "td_r": 50.,
                "eso_w0": 250., "k1": 30., "k2": 17., "alpha1": .75, "alpha2": .7, "delta": .02}
    if len(controller_settings) != 4 or any(
        any(settings.get(key) != value for key, value in expected.items()) for settings in controller_settings
    ):
        raise ValueError(f"{path}: controller settings differ from this model; audit separately")

    result = {
        "file": str(path),
        "csv_sha256": csv_sha256,
        "metadata_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        "recorder_status": status,
        "condition": metadata.get("condition"),
        "metadata": {
            key: metadata.get(key)
            for key in (
                "schema_version",
                "command_domain",
                "excitation_mode",
                "torque_calibrated",
                "min_angle_deg",
                "max_angle_deg",
                "slow_duration_s",
                "fast_duration_s",
                "ultra_fast_frequency_hz",
                "ultra_fast_duration_s",
                "step_enable",
                "step_duration_s",
                "step_cycles",
            )
        },
        "rows": 0,
        "bad_rows": 0,
        "stage_counts": defaultdict(int),
        "stage_transitions": [],
        "elapsed_s": _Stats(),
        "sample_period_s": _Stats(),
        "frequency_hz": _Stats(),
        "joints": {},
        "step_rows": [],
        "ground_load_bins": {},
    }
    for joint in JOINTS:
        result["joints"][joint] = {
            "signals": {
                signal: _Stats()
                for signal in (
                    "physical_angle",
                    "physical_velocity",
                    "target_physical_angle",
                    "current_raw",
                    "command_current_raw",
                    "torque_proxy",
                    "torque_per_current_raw",
                    "feedback_age_s",
                )
            },
            "angle_error": _Stats(),
            "angle_error_abs": _Stats(),
            "proxy_ratio": _Stats(),
            "stage": _stage_stats(),
            "feedback_period_s": _Stats(),
            "command_period_s": _Stats(),
            "last_feedback_time_s": None,
            "last_command_time_s": None,
            "last_feedback_sequence": None,
            "last_command_sequence": None,
            "sequence_duplicates": {"feedback": 0, "command": 0},
        }
    result["imu"] = {signal: _Stats() for signal in ("roll", "pitch")}

    previous_host_time = None
    previous_stage = None
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        indices = {name: index for index, name in enumerate(header)}
        for required in (
            "/identification/host_time_s",
            "/identification/elapsed_s",
            "/identification/stage",
            "/identification/frequency_hz",
        ):
            if required not in indices:
                raise ValueError(f"{path}: missing required signal {required}")

        for row in reader:
            result["rows"] += 1
            if len(row) != len(header):
                result["bad_rows"] += 1
            host_time = _float(row, indices, "/identification/host_time_s")
            elapsed = _float(row, indices, "/identification/elapsed_s")
            stage_value = _float(row, indices, "/identification/stage")
            frequency = _float(row, indices, "/identification/frequency_hz")
            stage = str(int(stage_value)) if _finite(stage_value) else "nan"
            result["stage_counts"][stage] += 1
            if previous_stage is not None and stage != previous_stage and len(result["stage_transitions"]) < 32:
                result["stage_transitions"].append({"row": result["rows"], "elapsed_s": elapsed, "from": previous_stage, "to": stage})
            previous_stage = stage
            result["elapsed_s"].add(elapsed)
            result["frequency_hz"].add(frequency)
            if _finite(host_time) and _finite(previous_host_time):
                result["sample_period_s"].add(host_time - previous_host_time)
            previous_host_time = host_time if _finite(host_time) else previous_host_time
            for signal in ("roll", "pitch"):
                result["imu"][signal].add(_float(row, indices, f"/chassis/imu/{signal}"))

            for joint in JOINTS:
                item = result["joints"][joint]
                angle = _float(row, indices, _joint_signal(joint, "physical_angle"))
                target = _float(row, indices, _joint_signal(joint, "target_physical_angle"))
                velocity = _float(row, indices, _joint_signal(joint, "physical_velocity"))
                current = _float(row, indices, _joint_signal(joint, "current_raw"))
                command_current = _float(row, indices, _joint_signal(joint, "command_current_raw"))
                torque_proxy = _float(row, indices, _joint_signal(joint, "torque"))
                ratio = _float(row, indices, _joint_signal(joint, "torque_per_current_raw"))
                feedback_time = _float(row, indices, _joint_signal(joint, "feedback_time_s"))
                command_time = _float(row, indices, _joint_signal(joint, "command_prepare_time_s"))
                feedback_age = _float(row, indices, _joint_signal(joint, "feedback_age_s"))
                sequence = _float(row, indices, _joint_signal(joint, "feedback_sequence"))
                command_sequence = _float(row, indices, _joint_signal(joint, "command_sequence"))

                for signal, value in (
                    ("physical_angle", angle),
                    ("physical_velocity", velocity),
                    ("target_physical_angle", target),
                    ("current_raw", current),
                    ("command_current_raw", command_current),
                    ("torque_proxy", torque_proxy),
                    ("torque_per_current_raw", ratio),
                    ("feedback_age_s", feedback_age),
                ):
                    item["signals"][signal].add(value)
                if _finite(angle) and _finite(target):
                    item["angle_error"].add(angle - target)
                    item["angle_error_abs"].add(abs(angle - target))
                if _finite(current) and abs(current) > 1e-9 and _finite(torque_proxy):
                    item["proxy_ratio"].add(torque_proxy / current)
                _record_period(item, "feedback", feedback_time, sequence)
                _record_period(item, "command", command_time, command_sequence)

                stage_item = item["stage"].setdefault(stage, {"rows": 0, **{name: _Stats() for name in ("angle", "target", "velocity", "current", "command_current", "error", "current_saturated")}})
                stage_item["rows"] += 1
                if _finite(command_current):
                    stage_item["current_saturated"].add(float(abs(command_current) >= RAW_CURRENT_LIMIT - 0.5))
                for name, value in (("angle", angle), ("target", target), ("velocity", velocity), ("current", current), ("command_current", command_current), ("error", angle - target if _finite(angle) and _finite(target) else float("nan"))):
                    stage_item[name].add(value)

                if metadata.get("step_enable") and stage == "10" and _finite(elapsed):
                    result["step_rows"].append((elapsed, joint, angle, target, velocity, current, command_current))

                # The grounded/suspended comparison below only uses slow/fast
                # motion near zero velocity, where it is a load diagnostic.
                if stage in ("3", "7") and _finite(angle) and _finite(velocity) and abs(velocity) < 0.03 and _finite(current):
                    bin_index = min(11, max(0, int((math.degrees(angle) - ANGLE_MIN_DEG) / (ANGLE_MAX_DEG - ANGLE_MIN_DEG) * 12)))
                    result["ground_load_bins"].setdefault(joint, defaultdict(list))[str(bin_index)].append(current)

    for joint, item in result["joints"].items():
        item["signals"] = {key: value.as_dict() for key, value in item["signals"].items()}
        item["angle_error"] = item["angle_error"].as_dict()
        item["angle_error_abs"] = item["angle_error_abs"].as_dict()
        item["proxy_ratio"] = item["proxy_ratio"].as_dict()
        item["feedback_period_s"] = item["feedback_period_s"].as_dict()
        item["command_period_s"] = item["command_period_s"].as_dict()
        item["stage"] = {
            stage: {
                key: value.as_dict() if isinstance(value, _Stats) else value
                for key, value in stats.items()
            }
            for stage, stats in item["stage"].items()
            if stats["rows"]
        }
    result["stage_counts"] = dict(result["stage_counts"])
    result["elapsed_s"] = result["elapsed_s"].as_dict()
    result["sample_period_s"] = result["sample_period_s"].as_dict()
    result["frequency_hz"] = result["frequency_hz"].as_dict()
    result["imu"] = {key: value.as_dict() for key, value in result["imu"].items()}
    return result


def _step_metrics(audit: list[dict]) -> dict:
    """Per-recording step metrics; no window may cross the next target jump."""
    result = {joint: {"transitions": [], "stable_tail": []} for joint in JOINTS}
    for file_result in audit:
        by_joint = defaultdict(list)
        for elapsed, joint, angle, target, velocity, current, command_current in file_result.get("step_rows", []):
            if all(_finite(v) for v in (elapsed, angle, target, velocity, current, command_current)):
                by_joint[joint].append((elapsed, math.degrees(angle), math.degrees(target),
                                        velocity, current, command_current))
        for joint, values in by_joint.items():
            jumps = [i for i in range(1, len(values)) if abs(values[i][2] - values[i - 1][2]) > 20.0]
            for number, index in enumerate(jumps):
                end = jumps[number + 1] if number + 1 < len(jumps) else len(values)
                t0, _, new_target, *_ = values[index]
                old_target = values[index - 1][2]
                direction = 1. if new_target > old_target else -1.
                window = values[index:end]
                crossings = {}
                for fraction in (0.1, 0.5, 0.9, 0.98):
                    threshold = old_target + fraction * (new_target - old_target)
                    for candidate in window:
                        if direction * (candidate[1] - threshold) >= 0:
                            crossings[str(fraction)] = candidate[0] - t0
                            break
                rise = crossings["0.9"] - crossings["0.1"] if "0.1" in crossings and "0.9" in crossings else None
                result[joint]["transitions"].append({
                    "file": file_result["file"], "start_elapsed_s": t0,
                    "from_deg": old_target, "to_deg": new_target,
                    "rise_crossing_s": crossings, "rise_10_90_s": rise,
                    "overshoot_deg": max(0., max(direction * (v[1] - new_target) for v in window)),
                    "peak_velocity_rad_s": max(abs(v[3]) for v in window),
                    "current_saturation_fraction": statistics.mean(abs(v[5]) >= RAW_CURRENT_LIMIT - .5 for v in window),
                })
                # Stable last two seconds, retaining a 0.5 s margin before the
                # next transition. These are closed-loop fluctuations, not
                # isolated encoder-noise or current-loop-noise measurements.
                tail = [v for v in window if max(t0 + 2.5, window[-1][0] - 2.) <= v[0] <= window[-1][0] - .5]
                if len(tail) > 1:
                    result[joint]["stable_tail"].append({
                        "file": file_result["file"], "target_deg": new_target,
                        "angle_std_deg": statistics.pstdev(v[1] for v in tail),
                        "command_current_std_raw": statistics.pstdev(v[5] for v in tail),
                        "tracking_error_mean_deg": statistics.mean(v[1] - v[2] for v in tail),
                    })
    return result


def _merge_ground_load(audit: list[dict]) -> dict:
    bins: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for file_result in audit:
        condition = file_result.get("condition")
        if condition not in ("grounded", "suspended"):
            continue
        for joint, values in file_result.get("ground_load_bins", {}).items():
            for index, currents in values.items():
                bins[joint][index][condition].extend(currents)
    result = {}
    for joint, joint_bins in bins.items():
        result[joint] = {}
        for index, conditions in joint_bins.items():
            if not conditions.get("grounded") or not conditions.get("suspended"):
                continue
            grounded = statistics.median(conditions["grounded"])
            suspended = statistics.median(conditions["suspended"])
            result[joint][index] = {
                "grounded_median_current_raw": grounded,
                "suspended_median_current_raw": suspended,
                "grounded_minus_suspended_current_raw": grounded - suspended,
                "grounded_count": len(conditions["grounded"]),
                "suspended_count": len(conditions["suspended"]),
            }
    return result


def build_model(audit: list[dict]) -> dict:
    feedback_periods = []
    command_periods = []
    feedback_ages = []
    angle_errors = []
    for file_result in audit:
        status = file_result.get("recorder_status") or {}
        if not status.get("completed") or status.get("write_failed") or status.get("dropped_samples", 0):
            continue
        for joint_result in file_result["joints"].values():
            feedback_periods.append(joint_result["feedback_period_s"].get("p50"))
            command_periods.append(joint_result["command_period_s"].get("p50"))
            feedback_ages.append(joint_result["signals"]["feedback_age_s"].get("p50"))
            angle_errors.append(joint_result["angle_error_abs"].get("p50"))
    feedback_age = _median_or_default(feedback_ages, float("nan"))
    feedback_period = _median_or_default(feedback_periods, float("nan"))
    command_period = _median_or_default(command_periods, float("nan"))
    if not all(_finite(v) for v in (feedback_age, feedback_period, command_period)):
        raise ValueError("complete, loss-free logs with usable timing are required")
    return {
        "schema_version": 1,
        "model_name": "deformable_real2sim_current_domain_v1",
        "provenance": {
            "source_schema_version": 2,
            "source_command_domain": "angle_reference",
            "source_torque_calibrated": False,
            "source_condition_pairs": sorted({item["condition"] for item in audit}),
            "sources": [{key: item[key] for key in ("file", "csv_sha256", "metadata_sha256", "rows", "recorder_status")} for item in audit],
            "note": "Closed-loop sweep identifies the equivalent current-domain servo. It does not independently identify motor inertia, friction, backlash, or calibrated shaft torque.",
        },
        "angle": {
            "convention": "physical_angle_relative_to_horizontal",
            "min_deg": ANGLE_MIN_DEG,
            "max_deg": ANGLE_MAX_DEG,
            "urdf_q_rad": "q = max_physical_angle - physical_angle",
        },
        "motor_source_constants": {
            "motor_type": "MG5010Ei36",
            "protocol_current_max_a": PROTOCOL_CURRENT_MAX_A,
            "raw_current_max_count": RAW_CURRENT_LIMIT,
            "torque_constant_nominal": MOTOR_TORQUE_CONSTANT_NOMINAL,
            "reduction_ratio": MOTOR_REDUCTION_RATIO,
            "reported_proxy_nm_per_raw_count": PROTOCOL_CURRENT_MAX_A / RAW_CURRENT_LIMIT * MOTOR_TORQUE_CONSTANT_NOMINAL * MOTOR_REDUCTION_RATIO,
            "reported_proxy_is_shaft_torque": False,
        },
        "simulation_effort_mapping": {
            "effort_unit": "N*m at the URDF actuated leg joint",
            "effort_limit_nm": SIM_EFFORT_LIMIT_NM,
            "nominal_torque_per_current_raw_nm": SIM_EFFORT_LIMIT_NM / RAW_CURRENT_LIMIT,
            "mapping": "clamp(current_raw, +/-raw_current_max) / raw_current_max * effort_limit_nm",
            "calibration_status": "uncalibrated_simulation_prior; 25 Nm limit does not determine Nm/count",
        },
        "controller_domain": {
            "output_domain": "current_raw",
            "controller_output_to_current_raw": 5.74635241301908,
            "dt_s": 0.001,
            "b0": -1.0,
            "observed_feedback_age_s": {"median": feedback_age, "recommended_delay_steps": max(1, round(feedback_age / 0.001))},
            "observed_feedback_period_s": feedback_period,
            "observed_command_period_s": command_period,
            "period_method": "positive timestamp delta / positive sequence delta; interval averages, not per-packet jitter",
            "sweep_reference": "finite target velocity bypasses TD; RL targets use rate limiter plus TD",
        },
        "identified_metrics": {
            "median_abs_angle_tracking_error_rad": _median_or_default(angle_errors, 0.01),
            "command_period_steps": max(1, round(command_period / 0.001)),
            "recommended_feedback_delay_steps": max(1, round(feedback_age / 0.001)),
            "note": "Closed-loop response only. Tracking error includes holds and stress tests; see per-stage audit before fitting.",
        },
        "unidentified_priors": {
            "command_transport_delay_steps_range": [0, 1],
            "motor_current_lag_tau_s": 0.003,
            "backlash_range_rad": [0.0, 0.01],
            "coulomb_friction_range_nm": [0.03, 0.20],
            "viscous_friction_range_nm_s": [0.005, 0.05],
            "note": "Not measured fits; amplitude and wear parameters are not independently identifiable from these closed-loop logs.",
        },
        "joint_noise_priors": {
            joint: {
                "angle_noise_std_rad_range": [0.0004, 0.0025],
                "velocity_noise_std_rad_s_range": [0.002, 0.02],
                "current_noise_std_raw_range": [4.0, 80.0],
            }
            for joint in JOINTS
        },
    }


def _json_ready(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", nargs="+", required=True, help="CSV paths; repeat or use a shell glob")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output compact JSON model path")
    parser.add_argument("--report", type=pathlib.Path, help="Optional full audit report (can be large)")
    args = parser.parse_args()
    paths = list(dict.fromkeys(pathlib.Path(value).expanduser().resolve() for group in args.csv for value in group))
    if not paths:
        raise SystemExit("at least one --csv is required")
    audit = [_audit_file(path) for path in paths]
    model = build_model(audit)
    step_metrics = _step_metrics(audit)
    compact_step = {}
    for joint, metrics in step_metrics.items():
        transitions = metrics.get("transitions", [])
        rise_times = [item["rise_10_90_s"] for item in transitions]
        overshoots = [item["overshoot_deg"] for item in transitions]
        stable = metrics.get("stable_tail", [])
        compact_step[joint] = {
            "transition_count": len(transitions),
            "rise_10_90_s_median": _median_or_default(rise_times, float("nan")),
            "overshoot_deg_median": _median_or_default(overshoots, float("nan")),
            "stable_angle_std_deg_median": _median_or_default(
                [item["angle_std_deg"] for item in stable], float("nan")
            ),
            "stable_command_current_std_raw_median": _median_or_default(
                [item["command_current_std_raw"] for item in stable], float("nan")
            ),
        }
    model["step_response"] = compact_step
    report = {
        "model": model,
        "files": audit,
        "step_metrics": step_metrics,
        "grounded_minus_suspended_current_raw": _merge_ground_load(audit),
    }
    report["model"]["provenance"]["source_files"] = [str(path) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(model), ensure_ascii=False, indent=2) + "\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "files": len(paths),
        "rows": sum(item["rows"] for item in audit),
        "feedback_age_median_s": model["controller_domain"]["observed_feedback_age_s"]["median"],
        "feedback_delay_steps": model["controller_domain"]["observed_feedback_age_s"]["recommended_delay_steps"],
        "effort_per_raw_count_nm": model["simulation_effort_mapping"]["nominal_torque_per_current_raw_nm"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
