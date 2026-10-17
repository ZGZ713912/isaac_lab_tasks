"""Prepare training-only preservation replay that admits motion-stable states.

Initial/command-ramp sensor states remain protected. Settled source states with
large tracking error do not impose the same hard preservation constraint. This
is a candidate training dataset, not a policy or a physics acceptance result.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import yaml

from deformable_mlp_motion_audit import DEFAULT_LIMITS, steady_samples
from deformable_mlp_replay import SCENARIOS, build as validate_sources


def source_control_contract(report, data):
    if (report["real2sim"]["mode"] != "fixed_parameters_zero_temporal_noise"
            or report["real2sim"]["observation_version"] != "current_fraction_v2"
            or report["command_frame"] != "body"):
        raise ValueError("Motion admission needs nominal body-frame V3 sensor provenance")
    source = Path(str(data["controller_checkpoint"]))
    params = source.parent/"params"/"env.yaml"
    env = yaml.load(params.read_text(),Loader=yaml.BaseLoader)
    slew = [float(env[key]) for key in ("drive_linear_acceleration_limit","drive_yaw_acceleration_limit")]
    if slew != [2.,4.]:
        raise ValueError("Candidate admission expects the actual collection's 2/4 command slew contract")
    return dict(controller_params=str(params),controller_params_sha256=hashlib.sha256(params.read_bytes()).hexdigest(),
                command_frame="body",drive_slew_limits=slew)


def eligible_states(critic, step_dt, slew_limits=(2., 4.)):
    if (critic.ndim != 3 or critic.shape[-1] != 40 or not np.isfinite(critic).all()
            or not np.isfinite(step_dt) or step_dt <= 0):
        raise ValueError("Admission requires finite training critic states and a positive cadence")
    command = critic[..., 1:4].copy()
    command[..., 2] *= 4.  # ABI yaw-command scale is .25.
    time = (np.arange(len(critic)) + 1) * step_dt
    age = np.broadcast_to(time[:, None], critic.shape[:2])
    steady = steady_samples(time, age, command, DEFAULT_LIMITS["transition_settle_s"], slew_limits)
    linear_error = np.linalg.norm(critic[..., 32:34] - command[..., :2], axis=-1)
    # This is the nominal collection's IMU measurement, not privileged angular
    # velocity. Linear velocity and wheel loads are training-only critic terms.
    yaw_error = np.abs(2 * critic[..., 6] - command[..., 2])
    parked = np.all(np.abs(command) < 1.e-6, axis=-1)
    linear_limit = np.where(parked, DEFAULT_LIMITS["parking_speed_p95_m_s"],
                            DEFAULT_LIMITS["linear_error_p95_m_s"])
    valid_motion = (linear_error <= linear_limit) & (yaw_error <= DEFAULT_LIMITS["yaw_error_p95_rad_s"])
    return ~steady | valid_motion, steady


def select_admitted(observations, critic, scenarios, dt, rng, budget, excluded, slew_limits=(2., 4.)):
    if observations.shape != (*critic.shape[:2], 160) or not np.isfinite(observations).all():
        raise ValueError("Admission requires aligned finite H5 sensor and critic states")
    if budget <= 0 or len(scenarios) != observations.shape[1] or set(scenarios.tolist()) != SCENARIOS:
        raise ValueError("Admission requires all six scenarios and a positive sampling budget")
    eligible, steady = eligible_states(critic, dt, slew_limits)
    eligible[:, np.arange(len(scenarios)) % 4 == 3] = False
    eligible[:, np.isin(np.arange(len(scenarios)), excluded)] = False
    startup = np.arange(len(critic)) * dt < 2.
    pairs = []
    for case in sorted(SCENARIOS):
        for early in (True, False):
            mask = eligible & (scenarios == case)[None] & (startup == early)[:, None]
            candidates = np.argwhere(mask)
            if not len(candidates):
                raise ValueError(f"No admitted training states for {case}, startup={early}")
            ids = rng.choice(len(candidates), size=budget, replace=len(candidates) < budget)
            pairs.append(candidates[ids])
    pairs = np.concatenate(pairs)
    return observations[pairs[:, 0], pairs[:, 1]].astype(np.float32), pairs, eligible, steady


def build(sources, output_dir, seed=2422, per_stratum=256):
    if seed in (1234, 4321) or seed < 0 or per_stratum <= 0:
        raise ValueError("Use a separate training seed and positive sampling budget")
    if output_dir.exists():
        raise ValueError("Candidate replay output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Reuse the original exact source/checkpoint/physics/failed-vehicle audit.
    # The staging dataset is discarded, and never replaces current V16 replay.
    with tempfile.TemporaryDirectory(prefix="motion_replay_source_audit_", dir=output_dir.parent) as temporary:
        _, manifest = validate_sources(sources, Path(temporary)/"validated", seed, per_stratum)
    rng = np.random.default_rng(seed)
    rows, envs, grades, seeds, provenance = [], [], [], [], []
    for source in manifest["sources"]:
        path = Path(source["path"])
        report = json.loads(path.parent.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as data:
            control = source_control_contract(report,data)
            selected, ids, eligible, steady = select_admitted(
                data["policy_obs"], data["critic_obs"], data["env_scenarios"],
                float(data["step_dt_s"]), rng, per_stratum, source["excluded_failed_vehicles"])
            # The original report excludes whole vehicles that reset; this
            # lets collection time stand for episode age in retained vehicles.
            retained = set(np.unique(ids[:, 1]).tolist())
            for case,row in report["results"]["POLICY"].items():
                vehicles = np.flatnonzero(data["env_scenarios"] == case)
                if any(vehicles[e["env_id"]] in retained and (e["terminated"] or e["timeout"])
                       for e in row["episodes"]):
                    raise ValueError("Admitted source vehicle contains a reset")
            rows.append(selected);envs.append(ids[:, 1])
            grades.append(np.full(len(ids), source["grade_deg"], np.float32))
            seeds.append(np.full(len(ids), source["seed"], np.int64))
            item = source.copy()
            original_eligible,_ = eligible_states(data["critic_obs"],float(data["step_dt_s"]))
            item.update(motion_admission_selected_settled_samples=int(steady[ids[:, 0], ids[:, 1]].sum()),
                        motion_admission_excluded_settled_states=int((steady & ~original_eligible).sum()),
                        source_control_contract=control)
            provenance.append(item)
    output_dir.mkdir()
    path = output_dir/"policy_obs.npz"
    np.savez_compressed(path, policy_obs=np.concatenate(rows),env_id=np.concatenate(envs),
                        grade_deg=np.concatenate(grades),seed=np.concatenate(seeds))
    manifest.update(sources=provenance,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    purpose="Candidate motion-admitted training sensor replay; frozen loaded actor supplies targets",
                    motion_admission=dict(thresholds=DEFAULT_LIMITS,drive_slew_limits=[2.,4.],
                                          protected_states="episode startup and command ramps, or settled motion-stable source states",
                                          evaluation_data_used=False,privileged_data="source critic only for training admission",
                                          counterfactual_reference_motion_guaranteed=False),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(manifest,indent=2)+"\n")
    (output_dir/"source.py").write_bytes(Path(__file__).read_bytes())
    return path,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",nargs="+",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=2422)
    parser.add_argument("--per-stratum",type=int,default=256)
    args=parser.parse_args()
    path,manifest=build(args.sources,args.output_dir.resolve(),args.seed,args.per_stratum)
    print(json.dumps(dict(path=str(path),samples=manifest["samples"],sha256=manifest["sha256"])))


if __name__ == "__main__":
    main()
