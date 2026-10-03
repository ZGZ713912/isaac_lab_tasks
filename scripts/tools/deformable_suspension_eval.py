"""Bounded suspension acceptance evaluation; importable without Isaac Sim.

Only the final JSON document goes to stdout. Simulator/runner diagnostics go to
stderr. No checkpoints, videos, reports or configuration files are written.
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


def command_at(scenario, step, steps):
    """World-frame vx/vy (m/s), yaw rate (rad/s); schedule never follows resets."""
    commands = {
        "static": (0.0, 0.0, 0.0), "forward": (1.0, 0.0, 0.0),
        "lateral": (0.0, 1.0, 0.0), "spin_positive": (0.0, 0.0, 2 * math.pi),
        "spin_negative": (0.0, 0.0, -2 * math.pi),
    }
    if scenario == "dynamic":
        phases = ((0.0, 0.0, 0.0), (0.5, 0.0, 2 * math.pi),
                  (-0.5, 0.0, -2 * math.pi), (0.0, 0.0, 0.0))
        return phases[min(3, 4 * step // steps)]
    return commands[scenario]


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


def summarize(episodes):
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
        "timeout_resets": sum(e["timeout"] for e in episodes),
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
    return p


def configure_grade(cfg, grade):
    """A long ramp isolates grade capability from periodic crests and cell resets."""
    if grade is None:
        return
    if not math.isfinite(grade) or not 0 <= grade <= 20:
        raise ValueError("Constant test grade must be finite and in [0, 20]")
    sub = cfg.terrain.terrain_generator.sub_terrains["periodic_slope"]
    sub.angle_range = (grade, grade)
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
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if cfg.action_contract_version != "minangle_residual_v2" or cfg.enable_chassis_servo:
        raise ValueError("Requires minangle_residual_v2 with chassis servo disabled")
    cfg.seed = args.seed
    cfg.external_cmd_override = True
    cfg.commands_world_frame = True
    cfg.policy_history_length = history
    cfg.observation_space = 32 * history
    cfg.events = None
    grade = getattr(args, "grade_deg", None)
    configure_grade(cfg, grade)
    # Eliminate stochastic observation streams whose RNG draw counts depend on resets.
    for field in ("encoder_noise_std", "encoder_bias_std", "gyro_noise_std", "gyro_bias_std", "gravity_noise_std"):
        setattr(cfg, field, 0.0)
    cfg.max_sensor_delay_steps = 0
    friction = sum(cfg.tire_friction_range) / 2
    cfg.tire_friction_range = (friction, friction)
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    if grade is not None:
        # Keep all reset cells in one uphill ramp, at least 7.5 m from its ends.
        u.scene.env_origins[:, 0] = 90.0 - u._profile_x_offset
        u.scene.env_origins[:, 1] = 4.0 * (torch.arange(u.num_envs, device=u.device) - (u.num_envs - 1) / 2)
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
                  "friction": friction, "command_frame": "world", "results": {}}
        output["terrain"] = {"kind": "constant_ramp" if grade is not None else "registered_task",
                             "grade_deg": grade, "boundary_resets": cfg.boundary_reset_enabled}
        modes = ("POLICY",) if policy and getattr(args, "policy_only", False) else (
            ("POLICY", "ZERO") if policy else ("ZERO",))
        for mode in modes:
            output["results"][mode] = {}
            for scenario_index, scenario in enumerate(args.scenarios):
                seed = args.seed + scenario_index * 1000003
                counts = [0] * u.num_envs
                ages = [0] * u.num_envs
                pending = [[] for _ in range(u.num_envs)]
                episodes = []
                reset_hash = hashlib.sha256()
                previous_target = u.leg_target.clone()

                def controlled_reset(ids):
                    # Match each env's nth reset, even when the other mode survives longer.
                    for env_id in ids.tolist():
                        local_seed = (seed + env_id * 1009 + counts[env_id] * 9176) % (2**31)
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
                            reset_hash.update(state.detach().cpu().numpy().tobytes())

                def finish(env_id, terminated, timeout, censored=False):
                    rows = pending[env_id]
                    n = len(rows)
                    contact = sum(s[3] for s in rows) / n if n else None
                    joint = sum(s[4] for s in rows) / n if n else None
                    tilt = distribution([s[2] for s in rows])
                    episodes.append({"env_id": env_id, "episode": counts[env_id] - 1,
                                     "steps": ages[env_id], "settled_samples": n,
                                     "terminated": bool(terminated), "timeout": bool(timeout),
                                     "censored_at_budget": censored, "all_contact_rate": contact,
                                     "contact_and_horizontal_rate": joint, "tilt_deg": tilt,
                                     "success": bool(not terminated and n and contact >= 0.98
                                                     and joint >= 0.95 and tilt["rms"] < 3
                                                     and tilt["abs_p95"] < 3), "samples": rows})
                    pending[env_id] = []
                    ages[env_id] = 0

                def capture_dones():
                    terminated, timeout = original_dones()
                    data = u.robot.data
                    if grade is not None:
                        phase = (data.body_link_pos_w[:, u._wheel_body_ids, 0] + u._profile_x_offset) % 80.0
                        if not ((phase >= 0.0) & (phase <= 20.0)).all():
                            raise RuntimeError("Constant-grade evaluation left the validated uphill ramp interior")
                    g = data.projected_gravity_b
                    norm = g.norm(dim=-1).clamp_min(1.e-9)
                    roll = torch.atan2(-g[:, 1], -g[:, 2]) * (180 / math.pi)
                    pitch = torch.asin((g[:, 0] / norm).clamp(-1, 1)) * (180 / math.pi)
                    tilt = torch.acos((-g[:, 2] / norm).clamp(-1, 1)) * (180 / math.pi)
                    load = u.wheel_normal_forces
                    contact = (load > cfg.wheel_contact_force_threshold).all(-1)
                    q = data.joint_pos[:, u._legs_idx]
                    low, high = 0.0, cfg.leg_target_upper_limit
                    target = u.leg_target
                    vel = data.root_link_lin_vel_w
                    limit = cfg.leg_target_rate_limit * u.step_dt
                    rows = torch.stack((roll, pitch, tilt, contact.float(),
                                        (contact & (tilt < 3.0)).float(), load.amin(-1), u.chassis_clearance,
                                        (data.applied_torque[:, u._legs_idx].abs() >= 0.99 * cfg.max_leg_torque).float().mean(-1),
                                        (data.applied_torque[:, u._wheels_idx].abs() >= 0.99 * cfg.wheel_torque_limit).float().mean(-1),
                                        ((q < low + .02) | (q > high - .02)).float().mean(-1),
                                        ((target < low + .02) | (target > high - .02)).float().mean(-1),
                                        (q - target).abs().mean(-1), u._last_wheel_slip.abs().mean(-1),
                                        vel[:, 0], vel[:, 1], data.root_ang_vel_b[:, 2],
                                        u.cmd_buf[:, 0], u.cmd_buf[:, 1], u.cmd_buf[:, 2],
                                        u.body_top_height,
                                        ((target - previous_target).abs() >= .99 * limit).float().mean(-1)), -1)
                    if not torch.isfinite(rows).all():
                        raise RuntimeError("Nonfinite physical evaluation metrics")
                    for env_id, (row, term, tout) in enumerate(zip(rows.cpu().tolist(), terminated.tolist(), timeout.tolist())):
                        ages[env_id] += 1
                        if ages[env_id] * u.step_dt > 0.5 + 1.e-9:
                            pending[env_id].append(row)
                        if term or tout:
                            finish(env_id, term, tout)
                    return terminated, timeout

                u._reset_idx = controlled_reset
                u._get_dones = capture_dones
                u.common_step_counter = 0
                u._reset_count.zero_()
                obs, _ = wrapped.reset()
                initial_hash = reset_hash.hexdigest()
                if mode == "ZERO" and policy:
                    paired = output["results"]["POLICY"][scenario]["initial_state_sha256"]
                    if initial_hash != paired:
                        raise RuntimeError("POLICY/ZERO initial states differ")
                with torch.no_grad():
                    for step in range(args.steps):
                        if not app.is_running():
                            raise RuntimeError("Simulator stopped before evaluation budget completed")
                        command = torch.tensor(command_at(scenario, step, args.steps), device=u.device)
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
                            print(f"EVAL_PROGRESS {mode} {scenario} {step + 1}/{args.steps}", file=sys.stderr, flush=True)
                for env_id in range(u.num_envs):
                    if ages[env_id]:
                        finish(env_id, False, False, censored=True)
                result = summarize(episodes)
                result["initial_state_sha256"] = initial_hash
                result["scenario_seed"] = seed
                result["acceptance"] = acceptance(result)
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
            if saved_env.get("action_contract_version") != "minangle_residual_v2":
                p.error("checkpoint action contract is not minangle_residual_v2")
        args.headless = True
        launcher = AppLauncher(args)
        try:
            report = evaluate(args, agent_cfg, history, launcher.app)
            report["agent_yaml"] = str(yaml_path.resolve()) if yaml_path else None
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
