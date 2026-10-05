"""Bounded suspension acceptance evaluation; importable without Isaac Sim.

Only the final JSON document goes to stdout. Simulator/runner diagnostics go to
stderr. Optional traces record measured state without changing the policy input.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys


DEFAULT_TASK = "Robotics-Deformable-Suspension-Rough-History-Transformer-v2"
SCENARIOS = ("static", "forward", "lateral", "spin_positive", "spin_negative", "dynamic")
TRACE_COLUMNS = (
    "roll_deg", "pitch_deg", "tilt_deg", "all_contact", "contact_and_horizontal",
    "min_load_n", "clearance_m", "leg_current_saturation", "wheel_torque_saturation",
    "q_limit_fraction", "target_limit_fraction", "tracking_error_rad", "abs_slip_m_s",
    "vx_world", "vy_world", "wz_body", "cmd_vx_world", "cmd_vy_world", "cmd_wz",
    "body_top_height_m", "target_rate_saturation", "linear_speed_error_m_s", "yaw_error_rad_s",
)


def command_at(scenario, step, steps, profile="stress"):
    """Requested-frame vx/vy (m/s), body yaw rate; schedule never follows resets."""
    commands = {
        "static": (0.0, 0.0, 0.0), "forward": (1.0, 0.0, 0.0),
        "lateral": (0.0, 1.0, 0.0), "spin_positive": (0.0, 0.0, 2 * math.pi),
        "spin_negative": (0.0, 0.0, -2 * math.pi),
    }
    if scenario == "dynamic":
        phases = ((0.0, 0.0, 0.0), (0.5, 0.0, 2 * math.pi),
                  (-0.5, 0.0, -2 * math.pi), (0.0, 0.0, 0.0))
        command = phases[min(3, 4 * step // steps)]
    else:
        command = commands[scenario]
    if profile == "play":
        return command[0] * .8, command[1] * .5, command[2] * (1.5 / (2 * math.pi))
    if profile != "stress":
        raise ValueError("Unknown command profile")
    return command


def linear_velocity_error(velocity_world, velocity_body, command, command_frame):
    if command_frame == "world":
        velocity = velocity_world
    elif command_frame == "body":
        velocity = velocity_body
    else:
        raise ValueError("Unknown command frame")
    return (velocity[..., :2] - command[..., :2]).norm(dim=-1)


def validate_agent(cfg, requested_history=None):
    """Never merge registry defaults into a checkpoint's saved agent config."""
    if not isinstance(cfg, dict):
        raise ValueError("Saved agent YAML must be a mapping")
    for key in ("policy", "algorithm", "num_steps_per_env", "obs_groups"):
        if key not in cfg:
            raise ValueError(f"Saved agent YAML missing {key}")
    for key in ("policy", "algorithm"):
        if not isinstance(cfg[key], dict) or not cfg[key].get("class_name"):
            raise ValueError(f"Saved {key}.class_name is required")
    history = cfg["policy"].get("history_length")
    if type(history) is not int or history not in (1, 4, 8):
        raise ValueError("Saved policy.history_length must be 1, 4 or 8")
    if requested_history is not None and requested_history != history:
        raise ValueError("--history must match saved policy.history_length")
    if cfg.get("class_name", "OnPolicyRunner") != "OnPolicyRunner":
        raise ValueError("This evaluator supports saved OnPolicyRunner configs only")
    return history


def distribution(values):
    if not values:
        return {"rms": None, "abs_p50": None, "abs_p95": None, "abs_p99": None, "abs_max": None}
    ordered = sorted(abs(x) for x in values)
    def quantile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {"rms": math.sqrt(sum(x * x for x in values) / len(values)),
            "abs_p50": quantile(0.5), "abs_p95": quantile(0.95),
            "abs_p99": quantile(0.99), "abs_max": ordered[-1]}


def summarize(episodes, command_frame="world"):
    samples = [sample for episode in episodes for sample in episode.pop("samples")]
    n = len(samples)
    # A pre-settle failure cannot vanish from the acceptance denominator.
    short_failures = sum((e["terminated"] or e["timeout"]) and e["settled_samples"] == 0 for e in episodes)
    def rate(column):
        return sum(s[column] for s in samples) / n if n else None
    def adjusted(column):
        return sum(s[column] for s in samples) / (n + short_failures) if n + short_failures else None
    result = {
        "settled_samples": n, "short_failed_episodes": short_failures,
        "all_contact_rate": rate(3), "contact_and_horizontal_rate": rate(4),
        "failure_adjusted_all_contact_rate": adjusted(3),
        "failure_adjusted_contact_and_horizontal_rate": adjusted(4),
        "roll_deg": distribution([s[0] for s in samples]),
        "pitch_deg": distribution([s[1] for s in samples]),
        "tilt_deg": distribution([s[2] for s in samples]),
        "min_load_n": min((s[5] for s in samples), default=None),
        "clearance_min_m": min((s[6] for s in samples), default=None),
        "leg_torque_saturation_rate": rate(7), "wheel_torque_saturation_rate": rate(8),
        "terminated_resets": sum(e["terminated"] for e in episodes),
        "physical_terminated_resets": sum(e.get("physical_terminated", e["terminated"]) for e in episodes),
        "timeout_resets": sum(e["timeout"] for e in episodes),
        "terrain_boundary_violations": sum(e.get("terrain_boundary", False) for e in episodes),
        "reset_events": sum(e["terminated"] or e["timeout"] for e in episodes),
        "episode_success_rate": sum(e.get("success", False) for e in episodes) / len(episodes) if episodes else 0.0,
        "episodes": episodes,
    }
    if samples and len(samples[0]) > 9:
        result.update({
            "q_limit_fraction": rate(9), "target_limit_fraction": rate(10),
            "target_tracking_error_rad": rate(11), "mean_abs_slip_m_s": rate(12),
            "mean_velocity_world": [rate(i) for i in (13, 14, 15)],
            "mean_command_world": [rate(i) for i in (16, 17, 18)],
            "body_top_height_m": distribution([s[19] for s in samples]),
            "leg_target_rate_saturation_rate": rate(20),
        })
    if samples and len(samples[0]) >= len(TRACE_COLUMNS):
        result["linear_speed_error_m_s"] = distribution([s[21] for s in samples])
        result["yaw_error_rad_s"] = distribution([s[22] for s in samples])
    if command_frame == "body" and "mean_command_world" in result:
        result["mean_command_body"] = result.pop("mean_command_world")
    return result


def acceptance(result):
    checks = {
        "settled_samples": result["settled_samples"] > 0,
        "all_contact_ge_0.98": (result["failure_adjusted_all_contact_rate"] or 0.0) >= 0.98,
        "joint_success_ge_0.95": (result["failure_adjusted_contact_and_horizontal_rate"] or 0.0) >= 0.95,
        "episode_success_ge_0.95": result["episode_success_rate"] >= 0.95,
        "no_terminated_resets": result["terminated_resets"] == 0,
        "tilt_rms_lt_3_deg": result["tilt_deg"]["rms"] is not None and result["tilt_deg"]["rms"] < 3.0,
        "tilt_p95_lt_3_deg": result["tilt_deg"]["abs_p95"] is not None and result["tilt_deg"]["abs_p95"] < 3.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--agent-yaml", type=Path, help="Default: checkpoint parent / params / agent.yaml")
    p.add_argument("--num_envs", "--num-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=600, help="Policy steps per scenario and mode, including settling")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--history", type=int, choices=(1, 4, 8), help="Baseline history, or assert checkpoint history")
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--strict", action="store_true", help="Exit 1 on POLICY acceptance failure (ZERO if no checkpoint)")
    p.add_argument("--policy-only", action="store_true", help="Skip ZERO when comparing matching checkpoints.")
    p.add_argument("--grade-deg", type=float,
                   help="Constant long ramp (0..20 deg); retains physical safety and disables cell-boundary resets.")
    p.add_argument("--trace-dir", type=Path,
                   help="Save per-step physical metrics and leg state to NPZ files for plots.")
    p.add_argument("--real2sim-mode", choices=("nominal", "randomized"), default="nominal",
                   help="Nominal removes noise for paired control; randomized retains training parameter/noise ranges.")
    p.add_argument("--batch-scenarios", action="store_true",
                   help="Run scenarios in parallel; --num_envs is per scenario and must be 16 for heading coverage.")
    p.add_argument("--command-profile", choices=("stress", "play"), default="stress",
                   help="Stress uses 1m/s and 2pi rad/s; play uses vx=.8, vy=.5, yaw=1.5.")
    p.add_argument("--command-frame", choices=("world", "body"), default="world",
                   help="Use body with the play profile to match GUI keyboard translation.")
    return p


def configure_grade(cfg, grade):
    """A long ramp isolates grade capability from periodic crests and cell resets."""
    if grade is None:
        return
    if not math.isfinite(grade) or not 0 <= grade <= 20:
        raise ValueError("Constant test grade must be finite and in [0, 20]")
    sub = cfg.terrain.terrain_generator.sub_terrains["periodic_slope"]
    sub.angle_range = (grade, grade)
    sub.angle_choices = None
    sub.segment_length = 20.0
    cfg.boundary_reset_enabled = False
    cfg.spawn_dir_jitter = False


def evaluate(args, agent_cfg, history, app):
    import gymnasium as gym
    import numpy as np
    import torch
    import agent_world  # noqa: F401
    import agent_tasks  # noqa: F401
    import agent_rl.rsl_rl.modules  # noqa: F401
    import agent_rl.rsl_rl.algorithms  # noqa: F401; registers DiagnosticPPO
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    batched = getattr(args, "batch_scenarios", False)
    total_envs = args.num_envs * (len(args.scenarios) if batched else 1)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=total_envs)
    from scripts.utils.deformable_checkpoint import validate_deformable_checkpoint
    if args.checkpoint:
        validate_deformable_checkpoint(cfg, args.checkpoint)
    if cfg.action_contract_version not in ("minangle_residual_v2","minangle_physical_v3") or cfg.enable_chassis_servo:
        raise ValueError("Requires a minangle residual contract with chassis servo disabled")
    cfg.seed = args.seed
    cfg.external_cmd_override = True
    command_frame = getattr(args, "command_frame", "world")
    cfg.commands_world_frame = command_frame == "world"
    cfg.policy_history_length = history
    cfg.observation_space = 32 * history
    cfg.events = None
    grade = getattr(args, "grade_deg", None)
    configure_grade(cfg, grade)
    randomized = getattr(args, "real2sim_mode", "nominal") == "randomized"
    if not randomized:
        for field in ("encoder_noise_std", "encoder_bias_std", "gyro_noise_std", "gyro_bias_std", "gravity_noise_std"):
            setattr(cfg, field, 0.0)
        cfg.max_sensor_delay_steps = 0
        if getattr(cfg, "real2sim_enabled", False):
            cfg.real2sim_randomize = False
            for field in ("real2sim_current_noise_std_range", "real2sim_angle_noise_std_range",
                          "real2sim_velocity_noise_std_range", "real2sim_current_sensor_noise_std_range"):
                setattr(cfg, field, (0., 0.))
        friction = sum(cfg.tire_friction_range) / 2
        cfg.tire_friction_range = (friction, friction)
    else:
        if not getattr(cfg, "real2sim_enabled", False):
            raise ValueError("randomized Real2Sim evaluation requires a Real2Sim task")
        cfg.real2sim_randomize = True
        friction = None
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    if grade is not None:
        # Keep all reset cells in one uphill ramp, at least 7.5 m from its ends.
        u.scene.env_origins[:, 0] = 90.0 - u._profile_x_offset
        spacing = min(4.0, 120.0 / max(1, u.num_envs - 1))
        u.scene.env_origins[:, 1] = spacing * (torch.arange(u.num_envs, device=u.device) - (u.num_envs - 1) / 2)
        u.scene.env_origins[:, 2] = 0.0
    original_dones = u._get_dones
    original_reset = u._reset_idx
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions") if agent_cfg else None)
        policy = None
        runner = None
        if agent_cfg:
            saved = copy.deepcopy(agent_cfg)
            saved["device"] = args.device
            runner = OnPolicyRunner(wrapped, saved, log_dir=None, device=args.device)
            runner.load(str(args.checkpoint.resolve()), load_optimizer=False, map_location=u.device)
            policy = runner.get_inference_policy(device=u.device)
        output = {"task": args.task, "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                  "seed": args.seed, "num_envs": args.num_envs, "steps": args.steps,
                  "step_dt_s": u.step_dt, "settle_s": 0.5, "history": history,
                  "friction": friction, "command_frame": command_frame, "results": {}}
        output["batch_scenarios"] = batched
        output["total_envs"] = u.num_envs
        output["command_profile"] = getattr(args, "command_profile", "stress")
        output["policy_preprocessing"] = {
            "previous_action_pair_filter": bool(agent_cfg["policy"].get("previous_action_pair_filter", False))
            if agent_cfg else False,
        }
        output["terrain"] = {"kind": "constant_ramp" if grade is not None else "registered_task",
                             "grade_deg": grade, "boundary_resets": cfg.boundary_reset_enabled}
        if getattr(cfg, "real2sim_enabled", False):
            output["real2sim"] = {"mode": "training_randomization" if randomized else "fixed_parameters_zero_temporal_noise",
                                  "observation_version": cfg.real2sim_observation_version,
                                  "model_sha256": hashlib.sha256(Path(cfg.real2sim_model_path).read_bytes()).hexdigest(),
                                  "shaft_torque_calibrated": False}
        modes = ("POLICY",) if policy and getattr(args, "policy_only", False) else (
            ("POLICY", "ZERO") if policy else ("ZERO",))
        for mode in modes:
            output["results"][mode] = {}
            groups = [args.scenarios] if batched else [[s] for s in args.scenarios]
            for group in groups:
                seeds = {s: args.seed + args.scenarios.index(s) * 1000003 for s in group}
                case_names = [group[i // args.num_envs] for i in range(u.num_envs)]
                # Reset draws are forked per env below; temporal noise has the
                # same time-indexed stream for both modes, independent of resets.
                torch.manual_seed(seeds[group[0]])
                counts = [0] * u.num_envs
                ages = [0] * u.num_envs
                pending = [[] for _ in range(u.num_envs)]
                episodes = []
                reset_hash = {s: hashlib.sha256() for s in group}
                previous_target = u.leg_target.clone()
                trace_dir = getattr(args, "trace_dir", None)
                trace_metrics, trace_q, trace_target, trace_current, trace_age, trace_dones, trace_actions, trace_boundary, trace_positions = (
                    [], [], [], [], [], [], [], [], []
                )
                trace_loads, trace_roll, trace_wheel_effort, trace_wheel_speed, trace_wheel_target = [], [], [], [], []

                def controlled_reset(ids):
                    # Match each env's nth reset, even when the other mode survives longer.
                    for env_id in ids.tolist():
                        case = case_names[env_id]
                        local_seed = (seeds[case] + (env_id % args.num_envs) * 1009
                                      + counts[env_id] * 9176) % (2**31)
                        devices = [torch.device(u.device).index or 0] if str(u.device).startswith("cuda") else []
                        with torch.random.fork_rng(devices=devices):
                            torch.manual_seed(local_seed)
                            # VecEnv reset may be called from a wrapper inference
                            # context; simulator state writes require normal tensors.
                            with torch.inference_mode(False):
                                original_reset(ids.new_tensor([env_id]))
                        counts[env_id] += 1
                        if counts[env_id] == 1:
                            state = torch.cat((u.robot.data.root_state_w[env_id], u.robot.data.joint_pos[env_id]))
                            reset_hash[case].update(state.detach().cpu().numpy().tobytes())

                def finish(env_id, terminated, timeout, censored=False, terrain_boundary=False, physical_terminated=None):
                    rows = pending[env_id]
                    n = len(rows)
                    contact = sum(s[3] for s in rows) / n if n else None
                    joint = sum(s[4] for s in rows) / n if n else None
                    tilt = distribution([s[2] for s in rows])
                    episodes.append({"env_id": env_id % args.num_envs, "scenario": case_names[env_id],
                                     "episode": counts[env_id] - 1,
                                     "steps": ages[env_id], "settled_samples": n,
                                     "terminated": bool(terminated), "timeout": bool(timeout),
                                     "physical_terminated": bool(terminated if physical_terminated is None else physical_terminated),
                                     "terrain_boundary": bool(terrain_boundary),
                                     "censored_at_budget": censored, "all_contact_rate": contact,
                                     "contact_and_horizontal_rate": joint, "tilt_deg": tilt,
                                     "success": bool(not terminated and n and contact >= 0.98
                                                     and joint >= 0.95 and tilt["rms"] < 3
                                                     and tilt["abs_p95"] < 3), "samples": rows})
                    pending[env_id] = []
                    ages[env_id] = 0

                def capture_dones():
                    terminated, timeout = original_dones()
                    physical_terminated = terminated.clone()
                    data = u.robot.data
                    terrain_boundary = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
                    if grade is not None:
                        phase = (data.body_link_pos_w[:, u._wheel_body_ids, 0] + u._profile_x_offset) % 80.0
                        terrain_boundary = ~((phase >= 0.0) & (phase <= 20.0)).all(-1)
                        # Leaving the constant-grade interior is a measured
                        # boundary failure. End that environment and keep the
                        # remaining scenarios running so a partial simulator
                        # run cannot be mistaken for a completed benchmark.
                        terminated = terminated | terrain_boundary
                    g = data.projected_gravity_b
                    norm = g.norm(dim=-1).clamp_min(1.e-9)
                    roll = torch.atan2(-g[:, 1], -g[:, 2]) * (180 / math.pi)
                    pitch = torch.asin((g[:, 0] / norm).clamp(-1, 1)) * (180 / math.pi)
                    tilt = torch.acos((-g[:, 2] / norm).clamp(-1, 1)) * (180 / math.pi)
                    load = u.wheel_normal_forces
                    contact = (load > cfg.wheel_contact_force_threshold).all(-1)
                    q = data.joint_pos[:, u._legs_idx]
                    low, high = getattr(cfg,"leg_target_lower_limit",0.0), cfg.leg_target_upper_limit
                    if u._leg_actuator is not None:
                        saturated = u._leg_actuator.command_current_raw.abs() >= .99*u._leg_actuator.current_limit
                    else:
                        saturated = data.applied_torque[:,u._legs_idx].abs() >= .99*cfg.max_leg_torque
                    target = u.leg_target
                    vel = data.root_link_lin_vel_w
                    limit = cfg.leg_target_rate_limit * u.step_dt
                    rows = torch.stack((roll, pitch, tilt, contact.float(),
                                        (contact & (tilt < 3.0)).float(), load.amin(-1), u.chassis_clearance,
                                        saturated.float().mean(-1),
                                        (data.applied_torque[:, u._wheels_idx].abs() >= 0.99 * cfg.wheel_torque_limit).float().mean(-1),
                                        ((q < low + .02) | (q > high - .02)).float().mean(-1),
                                        ((target < low + .02) | (target > high - .02)).float().mean(-1),
                                        (q - target).abs().mean(-1), u._last_wheel_slip.abs().mean(-1),
                                        vel[:, 0], vel[:, 1], data.root_ang_vel_b[:, 2],
                                        u.cmd_buf[:, 0], u.cmd_buf[:, 1], u.cmd_buf[:, 2],
                                        u.body_top_height,
                                        ((target - previous_target).abs() >= .99 * limit).float().mean(-1),
                                        linear_velocity_error(vel, data.root_link_lin_vel_b,
                                                              u.cmd_buf, command_frame),
                                        (data.root_ang_vel_b[:, 2] - u.cmd_buf[:, 2]).abs()), -1)
                    if not torch.isfinite(rows).all():
                        raise RuntimeError("Nonfinite physical evaluation metrics")
                    if trace_dir is not None:
                        trace_metrics.append(rows.detach().cpu().numpy())
                        trace_q.append(q.detach().cpu().numpy().copy())
                        trace_target.append(target.detach().cpu().numpy().copy())
                        current = (u._leg_actuator.command_current_raw if u._leg_actuator is not None
                                   else data.applied_torque[:, u._legs_idx])
                        trace_current.append(current.detach().cpu().numpy().copy())
                        trace_age.append((np.asarray(ages) + 1) * u.step_dt)
                        trace_dones.append(torch.stack((terminated, timeout), -1).cpu().numpy())
                        trace_actions.append(actions.detach().cpu().numpy().copy())
                        trace_boundary.append(terrain_boundary.cpu().numpy())
                        trace_positions.append(data.root_link_pos_w.detach().cpu().numpy().copy())
                        trace_loads.append(load.detach().cpu().numpy().copy())
                        trace_roll.append(u._wheel_geometry_w()[-1].detach().cpu().numpy().copy())
                        trace_wheel_effort.append(u._wheel_tau.detach().cpu().numpy().copy())
                        trace_wheel_speed.append(data.joint_vel[:, u._wheels_idx].detach().cpu().numpy().copy())
                        trace_wheel_target.append(u._wheel_target.detach().cpu().numpy().copy())
                    for env_id, (row, term, tout, boundary, physical) in enumerate(zip(rows.cpu().tolist(), terminated.tolist(), timeout.tolist(), terrain_boundary.tolist(), physical_terminated.tolist())):
                        ages[env_id] += 1
                        if ages[env_id] * u.step_dt > 0.5 + 1.e-9 and not boundary:
                            pending[env_id].append(row)
                        if term or tout:
                            finish(env_id, term, tout, terrain_boundary=boundary, physical_terminated=physical)
                    return terminated, timeout

                u._reset_idx = controlled_reset
                u._get_dones = capture_dones
                u.common_step_counter = 0
                u._reset_count.zero_()
                # Explicit scenario resets reuse tick zero; discard the cached
                # observation so the first action sees this scenario's state.
                u._obs_tick = -1
                u._obs_cache = None
                obs, _ = wrapped.reset()
                initial_hashes = {s: reset_hash[s].hexdigest() for s in group}
                if mode == "ZERO" and policy:
                    for scenario in group:
                        paired = output["results"]["POLICY"][scenario]["initial_state_sha256"]
                        if initial_hashes[scenario] != paired:
                            raise RuntimeError("POLICY/ZERO initial states differ")
                with torch.no_grad():
                    for step in range(args.steps):
                        if not app.is_running():
                            raise RuntimeError("Simulator stopped before evaluation budget completed")
                        command = torch.tensor([command_at(s, step, args.steps, output["command_profile"])
                                                for s in case_names], device=u.device)
                        u.cmd_buf[:] = command
                        # Cached history is idempotent: patch current command, not a second history tick.
                        current = u._drive_cmd_b() * command.new_tensor((1.0, 1.0, 0.25))
                        obs["policy"][:, -31:-28] = current
                        obs["critic"][:, 1:4] = current
                        actions = policy(obs) if mode == "POLICY" else torch.zeros(u.num_envs, cfg.action_space, device=u.device)
                        if not torch.isfinite(actions).all():
                            raise RuntimeError("Nonfinite policy actions")
                        previous_target.copy_(u.leg_target)
                        obs, _, _, _ = wrapped.step(actions)
                        if (step + 1) % 100 == 0:
                            print(f"EVAL_PROGRESS {mode} {','.join(group)} {step + 1}/{args.steps}", file=sys.stderr, flush=True)
                for env_id in range(u.num_envs):
                    if ages[env_id]:
                        finish(env_id, False, False, censored=True)
                for case_index, scenario in enumerate(group):
                    result = summarize([e for e in episodes if e["scenario"] == scenario], command_frame)
                    result["initial_state_sha256"] = initial_hashes[scenario]
                    result["scenario_seed"] = seeds[scenario]
                    result["acceptance"] = acceptance(result)
                    if trace_dir is not None:
                        trace_dir.mkdir(parents=True, exist_ok=True)
                        path = trace_dir / f"{mode.lower()}_{scenario}.npz"
                        ids = slice(case_index * args.num_envs, (case_index + 1) * args.num_envs)
                        columns = tuple(c.replace("cmd_vx_world", "cmd_vx_body").replace("cmd_vy_world", "cmd_vy_body")
                                        for c in TRACE_COLUMNS) if command_frame == "body" else TRACE_COLUMNS
                        np.savez_compressed(path, metrics=np.stack(trace_metrics)[:, ids],
                                            columns=np.asarray(columns), command_frame=np.asarray(command_frame),
                                            time_s=(np.arange(len(trace_metrics)) + 1) * u.step_dt,
                                            episode_age_s=np.stack(trace_age)[:, ids], dones=np.stack(trace_dones)[:, ids],
                                            joint_q=np.stack(trace_q)[:, ids], leg_target=np.stack(trace_target)[:, ids],
                                            commanded_current_raw=np.stack(trace_current)[:, ids],
                                            raw_policy_actions=np.stack(trace_actions)[:, ids],
                                            terrain_boundary=np.stack(trace_boundary)[:, ids],
                                            root_pos_w=np.stack(trace_positions)[:, ids],
                                            wheel_normal_forces_n=np.stack(trace_loads)[:, ids],
                                            wheel_roll_direction_w=np.stack(trace_roll)[:, ids],
                                            wheel_motor_torque_nm=np.stack(trace_wheel_effort)[:, ids],
                                            wheel_speed_rad_s=np.stack(trace_wheel_speed)[:, ids],
                                            wheel_target_rad_s=np.stack(trace_wheel_target)[:, ids],
                                            physical_angle_zero_rad=np.asarray(getattr(cfg, "leg_physical_angle_zero",
                                                                                        math.radians(75))))
                        result["trace"] = str(path.resolve())
                    output["results"][mode][scenario] = result
                    print(f"EVAL_RESULT {mode} {scenario} contact={result['all_contact_rate']} "
                          f"joint={result['contact_and_horizontal_rate']} p95={result['tilt_deg']['abs_p95']}",
                          file=sys.stderr, flush=True)
        target = "POLICY" if policy else "ZERO"
        output["strict_target"] = target
        output["passed"] = all(r["acceptance"]["passed"] for r in output["results"][target].values())
        return output
    finally:
        u._get_dones = original_dones
        u._reset_idx = original_reset
        env.close()


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    for package in ("agent_world", "agent_tasks", "agent_rl"):
        sys.path.insert(0, str(root / "source" / package))
    # Imports and simulator startup also print diagnostics; keep stdout machine-readable.
    with redirect_stdout(sys.stderr):
        from isaaclab.app import AppLauncher
        p = parser()
        AppLauncher.add_app_launcher_args(p)
        args = p.parse_args()
        if args.steps <= 0 or args.num_envs <= 0 or args.seed < 0:
            p.error("steps and num_envs must be positive; seed must be nonnegative")
        if len(set(args.scenarios)) != len(args.scenarios):
            p.error("scenarios must be unique")
        if args.agent_yaml and not args.checkpoint:
            p.error("--agent-yaml requires --checkpoint")
        if args.policy_only and not args.checkpoint:
            p.error("--policy-only requires --checkpoint")
        if args.grade_deg is not None and (not math.isfinite(args.grade_deg) or not 0 <= args.grade_deg <= 20):
            p.error("--grade-deg must be finite and in [0, 20]")
        if args.grade_deg is not None and (args.num_envs > 24 or args.steps > 600):
            p.error("Constant-ramp budget supports at most 24 envs and 600 steps within the plane interior")
        if args.batch_scenarios and args.num_envs != 16:
            p.error("Batched scenarios require 16 envs per scenario to preserve reset heading strata")
        agent_cfg = None
        history = args.history or 8
        yaml_path = None
        if args.checkpoint:
            if not args.checkpoint.is_file():
                p.error("checkpoint does not exist")
            import yaml
            yaml_path = args.agent_yaml or args.checkpoint.parent / "params" / "agent.yaml"
            with yaml_path.open() as stream:
                agent_cfg = yaml.safe_load(stream)
            history = validate_agent(agent_cfg, args.history)
            env_path = args.checkpoint.parent / "params" / "env.yaml"
            if not env_path.is_file():
                p.error("checkpoint must retain params/env.yaml to verify its action contract")
            with env_path.open() as stream:
                # Isaac Lab YAML includes Python tags for config classes.
                saved_env = yaml.load(stream, Loader=yaml.BaseLoader)
            if saved_env.get("action_contract_version") not in ("minangle_residual_v2","minangle_physical_v3"):
                p.error("checkpoint action contract is not a supported minangle residual contract")
        args.headless = True
        launcher = AppLauncher(args)
        try:
            report = evaluate(args, agent_cfg, history, launcher.app)
            report["agent_yaml"] = str(yaml_path.resolve()) if yaml_path else None
            report["agent_yaml_sha256"] = hashlib.sha256(yaml_path.read_bytes()).hexdigest() if yaml_path else None
            # Kit shutdown can terminate Python before code after app.close runs.
            sys.stdout.write(json.dumps(report, allow_nan=False, sort_keys=True) + "\n")
            sys.stdout.flush()
        except BaseException:
            import traceback
            traceback.print_exc()
            raise
        finally:
            launcher.app.close()
    return 1 if args.strict and not report["passed"] else 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
