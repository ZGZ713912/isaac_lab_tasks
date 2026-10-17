"""Build training-only sensor replay for contact-sensitive recovered PPO."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCENARIOS = {"static", "forward", "lateral", "spin_positive", "spin_negative", "dynamic"}


def select_training_samples(observations, scenarios, step_dt, rng, per_stratum=256, excluded_vehicles=()):
    """Balance motion cases and startup without using held-out vehicles."""
    steps, envs, width = observations.shape
    if (width != 160 or not np.isfinite(observations).all() or step_dt <= 0
            or len(scenarios) != envs or set(scenarios.tolist()) != SCENARIOS):
        raise ValueError("Replay needs finite five-frame observations and all six motion cases")
    phase = np.arange(steps)*step_dt < 2.0
    pairs = []
    for scenario in sorted(SCENARIOS):
        vehicles = np.flatnonzero((np.arange(envs) % 4 != 3) & (scenarios == scenario)
                                 & ~np.isin(np.arange(envs), excluded_vehicles))
        for startup in (True, False):
            times = np.flatnonzero(phase == startup)
            if not len(times) or not len(vehicles):
                raise ValueError("Every replay stratum needs training vehicles and time samples")
            chosen = rng.choice(len(times)*len(vehicles), size=per_stratum,
                                replace=len(times)*len(vehicles) < per_stratum)
            pairs.append(np.stack((times[chosen//len(vehicles)], vehicles[chosen % len(vehicles)]), -1))
    ids = np.concatenate(pairs)
    return observations[ids[:, 0], ids[:, 1]].astype(np.float32), ids


def failed_vehicles(report, scenarios):
    """Report episode IDs are local to each scenario's vehicle batch."""
    rows = report["results"]["POLICY"]
    if (set(rows) != SCENARIOS
            or any(row["physical_terminated_resets"] or row["failure_adjusted_all_contact_rate"] < .98
                   for row in rows.values())):
        raise ValueError("Replay collections need supported, physically stable rollouts")
    excluded = set()
    for scenario, row in rows.items():
        vehicles = np.flatnonzero(scenarios == scenario)
        for episode in row["episodes"]:
            index = episode["env_id"]
            if type(index) is not int or not 0 <= index < len(vehicles):
                raise ValueError("Replay episode IDs differ from the scenario batch")
            if episode["terminated"] or episode["timeout"] or episode.get("terrain_boundary"):
                excluded.add(int(vehicles[index]))
    return sorted(excluded)


def build(sources, output_dir, seed=2419, per_stratum=256):
    if seed < 0 or seed in (1234, 4321) or per_stratum <= 0:
        raise ValueError("Use a separate training seed and a positive sample budget")
    rng = np.random.default_rng(seed)
    rows, ids, seeds, grades, provenance = [], [], [], [], []
    for directory in sources:
        directory = directory.resolve()
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text())
        if (status.get("status") != "completed" or "training" not in status.get("purpose", "")
                or type(status.get("seed")) is not int or status["seed"] in (1234, 4321)):
            raise ValueError("Replay source must be a completed training collection with a separate seed")
        for path in sorted(directory.glob("grade_*/*.npz")):
            report_path = directory / (path.parent.name+".json")
            report = json.loads(report_path.read_text())
            with np.load(path, allow_pickle=False) as data:
                if (int(data["seed"]) != status["seed"] or report["seed"] != status["seed"]
                        or data["policy_obs"].shape[0] != report["steps"]
                        or abs(float(data["step_dt_s"])-report["step_dt_s"]) > 1.e-9
                        or hashlib.sha256(Path(str(data["teacher_checkpoint"])).read_bytes()).hexdigest()
                        != status["teacher_sha256"]):
                    raise ValueError("Replay source provenance differs from the actual collection")
                excluded = failed_vehicles(report, data["env_scenarios"])
                obs, pairs = select_training_samples(data["policy_obs"], data["env_scenarios"],
                                                      float(data["step_dt_s"]), rng, per_stratum, excluded)
                grade = float(data["grade_deg"])
                rows.append(obs)
                ids.append(pairs[:, 1])
                seeds.append(np.full(len(obs), status["seed"], dtype=np.int64))
                grades.append(np.full(len(obs), grade, dtype=np.float32))
                provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                       seed=status["seed"], grade_deg=grade, samples=len(obs),
                                       status_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest(),
                                       report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                                       physics_sha256=str(data["real2sim_model_sha256"]),
                                       excluded_failed_vehicles=excluded))
    if not grades:
        raise ValueError("Replay requires actual collected sensor datasets")
    if set(np.concatenate(grades).tolist()) != {0., 5., 10., 17., 20.}:
        raise ValueError("Replay requires all five grades")
    if len({source["physics_sha256"] for source in provenance}) != 1:
        raise ValueError("Replay sources must use the same physical model")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "policy_obs.npz"
    np.savez_compressed(path, policy_obs=np.concatenate(rows), env_id=np.concatenate(ids),
                        seed=np.concatenate(seeds), grade_deg=np.concatenate(grades))
    manifest = dict(schema="deformable_sensor_replay_v1", status="completed",
                    partition="training_vehicles_env_id_mod4_ne3", sources=provenance,
                    purpose="Frozen loaded actor targets on training sensor states and neighborhoods only",
                    samples=sum(len(row) for row in rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    sample_seed=seed, per_stratum=per_stratum,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2)+"\n")
    (output_dir / "source.py").write_bytes(Path(__file__).read_bytes())
    return path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2419)
    parser.add_argument("--per-stratum", type=int, default=256)
    args = parser.parse_args()
    path, manifest = build(args.sources, args.output_dir.resolve(), args.seed, args.per_stratum)
    print(json.dumps(dict(path=str(path), samples=manifest["samples"], sha256=manifest["sha256"]), indent=2))


if __name__ == "__main__":
    main()
