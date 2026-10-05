"""Export TensorBoard scalars and plot measured training/evaluation comparisons.

Run labels refer to actual log directories. Evaluation files must contain the
completed JSON from deformable_suspension_eval.py, not a running status file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deformable-training-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TRAIN_PANELS = (
    ("Train/mean_reward", "Episode reward", 1),
    ("Train/mean_episode_length", "Episode length (policy steps)", 1),
    ("dynamic/all_contact", "Four-wheel contact (%)", 100),
    ("dynamic/residual_tilt_deg", "Mean body tilt (deg)", 1),
    ("dynamic/contact_and_horizontal", "Contact and tilt <3 deg (%)", 100),
    ("dynamic/speed_error", "Linear velocity error (m/s)", 1),
    ("dynamic/slip", "Mean longitudinal slip (m/s)", 1),
    ("dynamic/valid_episode_fraction", "Logged post-settle episode fraction (%)", 100),
    ("Policy/mean_noise_std", "Action noise standard deviation", 1),
    ("dynamic/baseline_extension_rad", "Lowest leg above 17 deg baseline (deg)", 180 / math.pi),
    ("dynamic/target_tracking_error", "Leg target tracking error (deg)", 180 / math.pi),
    ("dynamic/torque_saturation", "Leg current command saturation (%)", 100),
)


def labelled_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use label=/path/to/run-or-evaluation.json")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Label and path must be nonempty")
    return label, Path(path)


def save_figure(fig, path):
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def read_scalars(path):
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    result = {}
    for tag in accumulator.Tags()["scalars"]:
        # Keep the last event when a resumed run writes the same iteration.
        events = {event.step: event for event in accumulator.Scalars(tag)}
        result[tag] = [dict(step=step, wall_time=event.wall_time,
                            value=event.value if math.isfinite(event.value) else None)
                       for step, event in sorted(events.items())]
    return result


def smooth(values, window=15):
    if len(values) < window:
        return values
    half = window // 2
    return np.convolve(np.pad(values, (half, half), mode="edge"),
                       np.ones(window) / window, mode="valid")


def plot_training(runs, output):
    fig, axes = plt.subplots(4, 3, figsize=(14, 13), layout="constrained")
    records = {}
    palette = plt.get_cmap("tab20").colors
    colors = list(palette[::2]) + list(palette[1::2])
    for run_index, (label, path) in enumerate(runs):
        color = colors[run_index % len(colors)]
        scalars = read_scalars(path)
        snapshots = {}
        for name in ("agent.yaml", "env.yaml", "real2sim_model.json"):
            file = path / "params" / name
            if file.is_file():
                snapshots[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        records[label] = dict(run=str(path.resolve()), snapshots_sha256=snapshots, scalars=scalars)
        for ax, (tag, title, scale) in zip(axes.flat, TRAIN_PANELS):
            rows = [r for r in scalars.get(tag, []) if r["value"] is not None]
            if not rows:
                continue
            x = [r["step"] for r in rows]
            y = np.asarray([r["value"] * scale for r in rows])
            line, = ax.plot(x, smooth(y), label=label, linewidth=1.5, color=color)
            ax.plot(x, y, color=line.get_color(), alpha=.18, linewidth=.6)
    for ax, (_, title, _) in zip(axes.flat, TRAIN_PANELS):
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("PPO iteration")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8)
    fig.suptitle("Deformable fitted Real2Sim V3: training curves\n"
                 "Randomized terrain, faint raw curves, 15-point moving average\n"
                 "Stages change reward and terrain; reward curves are not acceptance comparisons",
                 fontsize=13)
    save_figure(fig, output / "training_curves.png")
    (output / "training_metrics.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return records


def plot_evaluations(evaluations, output):
    records, points = [], {}
    zero_conditions = {}
    for label, path in evaluations:
        report = json.loads(path.read_text())
        if "results" not in report or "terrain" not in report:
            raise ValueError(f"Not a completed evaluation: {path}")
        grade = report["terrain"]["grade_deg"]
        for mode, scenarios in report["results"].items():
            key = label if mode == "POLICY" else "ZERO"
            rows = list(scenarios.values())
            if not rows:
                raise ValueError(f"Empty evaluation in {path}")
            missing = [name for name, row in scenarios.items() if row["settled_samples"] == 0]
            point = dict(label=key, grade_deg=grade,
                         mode=report.get("real2sim", {}).get("mode", "legacy"),
                         seed=report["seed"], scenarios=list(scenarios),
                         steps=report["steps"], step_dt_s=report["step_dt_s"],
                         command_profile=report.get("command_profile", "stress"),
                         command_frame=report.get("command_frame", "world"),
                         real2sim_model_sha256=report.get("real2sim", {}).get("model_sha256"),
                         contact_min=min(r["failure_adjusted_all_contact_rate"] or 0 for r in rows),
                         horizontal_min=min(r["failure_adjusted_contact_and_horizontal_rate"] or 0 for r in rows),
                         tilt_p95_max=None if missing else max(r["tilt_deg"]["abs_p95"] for r in rows),
                         tilt_rms_max=None if missing else max(r["tilt_deg"]["rms"] for r in rows),
                         linear_speed_error_rms_max=(max(r["linear_speed_error_m_s"]["rms"] for r in rows)
                             if all(r.get("linear_speed_error_m_s", {}).get("rms") is not None for r in rows)
                             else None),
                         scenarios_without_settled_samples=missing,
                         terminated_resets=sum(r["terminated_resets"] for r in rows),
                         physical_terminated_resets=sum(r.get("physical_terminated_resets", r["terminated_resets"])
                                                        for r in rows),
                         terrain_boundary_violations=sum(r.get("terrain_boundary_violations", 0) for r in rows),
                         checkpoint=report.get("checkpoint"), report=str(path.resolve()),
                         report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            point["trace_summaries"] = {
                name: trace_summary(Path(row["trace"])) for name, row in scenarios.items() if row.get("trace")
            }
            point["static_baseline_extension_deg"] = point["trace_summaries"].get(
                "static", {}).get("mean_lowest_leg_extension_deg")
            condition = tuple(point[field] for field in (
                "mode", "seed", "steps", "step_dt_s", "command_profile", "command_frame", "real2sim_model_sha256"))
            if mode == "ZERO":
                short_mode = {'fixed_parameters_zero_temporal_noise': 'nominal',
                              'training_randomization': 'randomized'}.get(point['mode'], point['mode'])
                key = zero_conditions.setdefault(condition,
                    f"ZERO {point['steps'] * point['step_dt_s']:g}s/{point['command_profile']}/"
                    f"{point['command_frame']}/s{point['seed']}/{short_mode}")
                point["label"] = key
            previous = points.get(key, {}).get(grade)
            if previous is not None:
                previous_condition = tuple(previous[field] for field in (
                    "mode", "seed", "steps", "step_dt_s", "command_profile", "command_frame", "real2sim_model_sha256"))
                if previous_condition != condition or previous["scenarios"] != point["scenarios"]:
                    raise ValueError(f"Conflicting benchmark conditions for {key} at {grade} degrees; use distinct labels")
            records.append(point)
            if grade is not None:
                points.setdefault(key, {})[grade] = point
        for scenario, row in report["results"].get("POLICY", {}).items():
            if row.get("trace"):
                plot_trace(Path(row["trace"]), output / f"trace_{label}_{grade}deg_{scenario}.png",
                           label, grade, scenario)
    if points:
        fig, grid = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        axes = list(grid.flat)
        palette = plt.get_cmap("tab20").colors
        colors = list(palette[::2]) + list(palette[1::2])
        policy_index = 0
        for label, by_grade in points.items():
            is_zero = label.startswith("ZERO ")
            color = "#6b7280" if is_zero else colors[policy_index % len(colors)]
            policy_index += not is_zero
            grades = sorted(by_grade)
            for ax, field, scale in zip(axes, ("tilt_p95_max", "contact_min", "horizontal_min",
                                              "terminated_resets", "linear_speed_error_rms_max",
                                              "static_baseline_extension_deg"), (1, 100, 100, 1, 1, 1)):
                ax.plot(grades, [by_grade[g][field] * scale if by_grade[g][field] is not None else np.nan
                                 for g in grades], marker="x" if is_zero else "o",
                        linestyle="--" if is_zero else "-", color=color, label=label)
        axes[0].axhline(3, color="gray", linestyle="--", linewidth=1, label="Strict 3 deg")
        axes[1].axhline(98, color="gray", linestyle="--", linewidth=1)
        for ax, title in zip(axes, ("Worst-scenario tilt P95 (deg)", "Worst four-wheel contact (%)",
                                  "Worst contact and tilt <3 deg (%)",
                                  "Total physical or boundary resets (count)",
                                  "Worst linear speed error RMS (m/s)",
                                  "Static lowest leg above 17 deg baseline (deg)")):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Ramp grade (deg)")
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=8)
        fig.suptitle("Measured simulator benchmarks: worst of the listed command scenarios\n"
                     "Missing tilt points indicate a scenario failed before settling", fontsize=11)
        save_figure(fig, output / "slope_comparison.png")
    (output / "evaluation_metrics.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return records


def trace_summary(path, baseline_deg=17.):
    """Summarize measured leg posture after the same per-episode settling gate."""
    with np.load(path, allow_pickle=False) as data:
        ages = data["episode_age_s"]
        settled = ages > .5 + 1.e-9
        if "terrain_boundary" in data:
            settled &= ~data["terrain_boundary"]
        physical = np.rad2deg(float(data["physical_angle_zero_rad"]) - data["joint_q"])[settled]
        targets = np.rad2deg(float(data["physical_angle_zero_rad"]) - data["leg_target"])[settled]
        current = data["commanded_current_raw"][settled]
        action_cycle = {}
        if "raw_policy_actions" in data:
            actions = np.clip(data["raw_policy_actions"], -1., 1.)
            for lag in (1, 2):
                valid = settled[lag:] & settled[:-lag] & (ages[lag:] > ages[:-lag])
                differences = np.abs(actions[lag:] - actions[:-lag])[valid]
                action_cycle[f"mean_abs_action_difference_lag{lag}"] = (
                    float(differences.mean()) if len(differences) else None)
    if not len(physical):
        return dict(settled_samples=0, mean_physical_angle_deg=None,
                    mean_lowest_leg_extension_deg=None, lowest_leg_within_2deg_of_baseline_rate=None)
    lowest = physical.min(-1)
    return dict(settled_samples=len(physical), joint_order=["RF", "LF", "LB", "RB"],
                baseline_deg=baseline_deg, mean_physical_angle_deg=physical.mean(0).tolist(),
                mean_target_angle_deg=targets.mean(0).tolist(),
                mean_lowest_leg_extension_deg=float(np.maximum(0, lowest - baseline_deg).mean()),
                lowest_leg_within_2deg_of_baseline_rate=float((lowest <= baseline_deg + 2).mean()),
                mean_abs_command_current_raw=np.abs(current).mean(0).tolist(),
                peak_abs_command_current_raw=float(np.abs(current).max()), **action_cycle)


def plot_trace(path, output, label, grade, scenario):
    with np.load(path, allow_pickle=False) as data:
        metrics, times = data["metrics"], data["time_s"]
        columns = {str(name): i for i, name in enumerate(data["columns"])}
        angle = np.rad2deg(float(data["physical_angle_zero_rad"]) - data["joint_q"])
        current = data["commanded_current_raw"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, field, title in zip(axes.flat[:2], ("tilt_deg", "all_contact"),
                               ("Body tilt (deg)", "Four-wheel contact fraction")):
        values = metrics[:, :, columns[field]]
        ax.plot(times, values.mean(1), label="Environment mean")
        ax.fill_between(times, np.quantile(values, .1, axis=1), np.quantile(values, .9, axis=1), alpha=.2,
                        label="10-90% environment range")
        ax.set_ylabel(title)
    for j, name in enumerate(("RF", "LF", "LB", "RB")):
        axes[1, 0].plot(times, angle[:, 0, j], label=name)
        axes[1, 1].plot(times, current[:, 0, j], label=name)
    axes[1, 0].axhline(17, color="gray", linewidth=1, linestyle="--")
    axes[1, 0].set_ylabel("Leg angle, environment 0 (deg)")
    axes[1, 1].set_ylabel("Command current, environment 0 (counts)")
    for ax in axes.flat:
        ax.set_xlabel("Scenario time (s)")
        ax.grid(alpha=.2)
        ax.axvspan(0, .5, color="gray", alpha=.08)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(ncols=4, fontsize=8)
    fig.suptitle(f"{label}: {grade} deg, {scenario}\n"
                 "Full trace including initial settling; benchmark aggregates exclude first 0.5 s of each episode",
                 fontsize=11)
    save_figure(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=labelled_path, default=[])
    parser.add_argument("--evaluation", action="append", type=labelled_path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.run and not args.evaluation:
        parser.error("Provide at least one --run or --evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.run:
        plot_training(args.run, args.output_dir)
    if args.evaluation:
        plot_evaluations(args.evaluation, args.output_dir)
    print(str(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
