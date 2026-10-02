"""Isolate wheel traction on a constant ramp without a learned suspension policy.

Run in the isaaclab environment:
    python scripts/tools/deformable_grade_probe.py --device=cuda:0
"""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for package in ("agent_tasks", "agent_world"):
    sys.path.insert(0, str(ROOT / "source" / package))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--grade", type=float, default=4.0, help="Constant uphill angle in degrees.")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--strict", action="store_true", help="Check settled velocity, slip, contact and resets.")
parser.add_argument("--transitions", action="store_true", help="Test start, reversal and stop; use --steps=900 or more.")
parser.add_argument("--full-spin", action="store_true", help="Include both full-spin signs and combined world translation/spin.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
launcher = AppLauncher(args)

import gymnasium as gym
import torch

import agent_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from agent_tasks.direct.deformable_suspension import cfg_utils as du


def main():
    task = "Robotics-Deformable-Suspension-Rough-History-Transformer-v1"
    command_values = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0.6], [0, 0, 0]]
    if args.full_spin:
        command_values[3] = [0, 0, 2 * math.pi]
        command_values += [[0, 0, -2 * math.pi], [0.5, 0, 2 * math.pi], [-0.5, 0, -2 * math.pi]]
    count = len(command_values)
    cfg = parse_env_cfg(task, device=args.device, num_envs=count)
    sub = cfg.terrain.terrain_generator.sub_terrains["periodic_slope"]
    sub.angle_range = (args.grade, args.grade)
    sub.segment_length = 20.0
    cfg.seed = 42
    cfg.external_cmd_override = True
    cfg.commands_world_frame = True
    cfg.boundary_reset_enabled = False
    # Isolate drive dynamics from the suspension policy's height objective.
    cfg.height_termination_margin = 1.0
    cfg.episode_length_s = 30.0
    env = gym.make(task, cfg=cfg)
    try:
        u = env.unwrapped
        env.reset()
        data = u.robot.data
        q = data.joint_pos.clone()
        q[:, u._legs_idx] = du.Q_LOW
        q[:, u._ws_idx] = du.Q_LOW
        q[:, u._upper_idx] = -du.Q_LOW
        u.robot.write_joint_state_to_sim(q, torch.zeros_like(q))

        theta = math.radians(args.grade)
        pose = torch.zeros(count, 7, device=args.device)
        # Long ramps keep every robot on the same grade throughout the measurement.
        phase = 0.5 * sub.segment_length
        pose[:, 0] = 4.0 * sub.segment_length + phase - u._profile_x_offset
        pose[:, 1] = 4.0 * (torch.arange(count, device=args.device) - (count - 1) / 2.0)
        pose[:, 2] = (
            phase * math.tan(theta)
            + du.q_to_base_height(du.Q_LOW) / math.cos(theta)
            + 0.01
        )
        pose[:, 3] = math.cos(theta / 2.0)
        pose[:, 5] = -math.sin(theta / 2.0)
        u.robot.write_root_pose_to_sim(pose)
        u.robot.write_root_velocity_to_sim(torch.zeros(count, 6, device=args.device))
        u.leg_target[:] = du.Q_LOW
        u._leg_adrc.reset(torch.arange(count, device=args.device), q[:, u._legs_idx], u.leg_target)
        u._wheel_drive.reset(torch.arange(count, device=args.device))
        u._tire_deflection[:] = 0.0

        commands = torch.tensor(
            command_values,
            device=args.device, dtype=torch.float32,
        )
        stats = []
        phases = [[], [], []] if args.transitions else [[]]
        phase_steps = args.steps // len(phases)
        command_change = torch.zeros(2, device=args.device)
        previous_command = u._drive_command.clone()
        resets = torch.zeros(count, device=args.device)
        actions = torch.zeros(count, 4, device=args.device)
        for step in range(args.steps):
            phase_index = min(len(phases) - 1, step // phase_steps)
            requested = commands if phase_index == 0 else (-commands if phase_index == 1 else torch.zeros_like(commands))
            u.cmd_buf[:] = requested
            _, _, done, timeout, _ = env.step(actions)
            resets += (done | timeout).float()
            delta = u._drive_command - previous_command
            command_change[0] = torch.maximum(command_change[0], delta[:, :2].norm(dim=-1).max())
            command_change[1] = torch.maximum(command_change[1], delta[:, 2].abs().max())
            previous_command.copy_(u._drive_command)
            if step % phase_steps >= phase_steps // 2:
                data = u.robot.data
                sample = torch.cat((
                    # Commands describe base_link motion. The offset root COM
                    # orbits during spin; its periodic velocity is not drive jitter.
                    data.root_link_lin_vel_w[:, :2], data.root_ang_vel_b[:, 2:3],
                    u._last_servo_force[:, :2], u.wheel_normal_forces,
                    u._last_wheel_slip, data.joint_vel[:, u._wheels_idx],
                    u.body_top_height[:, None], u.chassis_clearance[:, None],
                ), dim=-1).clone()
                stats.append(sample)
                phases[phase_index].append(sample)
        means = torch.stack(stats).mean(0)
        print("GRADE_PROBE " + json.dumps({
            "grade_deg": args.grade,
            "steps": args.steps,
            "linear_velocity_reference": "base_link_origin_world",
            "commands": commands.tolist(),
            "columns": [
                "vx_w", "vy_w", "wz_b", "Fx_w", "Fy_w",
                "load1", "load2", "load3", "load4",
                "slip1", "slip2", "slip3", "slip4",
                "wheel1", "wheel2", "wheel3", "wheel4",
                "body_top_height", "chassis_clearance",
            ],
            "means": means.tolist(),
            "mean_abs_slip": torch.stack(stats)[..., 9:13].abs().mean((0, 2)).tolist(),
            "velocity_std": torch.stack(stats)[..., :3].std(0).tolist(),
            "phase_velocities": [torch.stack(samples).mean(0)[:, :3].tolist() for samples in phases],
            "phase_velocity_std": [torch.stack(samples)[..., :3].std(0).tolist() for samples in phases],
            "max_command_step": command_change.tolist(),
            "resets": resets.tolist(),
        }), flush=True)
        if args.strict:
            assert args.steps >= (900 if args.transitions else 600) and abs(args.grade) <= 8
            assert resets.sum() == 0
            for index, samples in enumerate(phases):
                reference = commands if index == 0 else (-commands if index == 1 else torch.zeros_like(commands))
                phase_mean = torch.stack(samples).mean(0)
                assert (phase_mean[:, :2] - reference[:, :2]).norm(dim=-1).max() < 0.08
                assert (phase_mean[:, 2] - reference[:, 2]).abs().max() < 0.05
                assert torch.stack(samples)[..., :3].std(0).max() < 0.03
            assert command_change[0] <= cfg.drive_linear_acceleration_limit * u.step_dt + 1e-5
            assert command_change[1] <= cfg.drive_yaw_acceleration_limit * u.step_dt + 1e-5
            assert torch.stack(stats)[..., 9:13].abs().mean((0, 2)).max() < 0.05
            assert torch.stack(stats)[..., 5:9].min() > cfg.wheel_contact_force_threshold
            if args.grade == 0:
                assert torch.stack(stats)[..., 17].max() <= cfg.max_body_top_height
                assert torch.stack(stats)[..., 18].min() >= 0.005
            print("GRADE_PROBE strict speed/slip/contact/smoothness checks passed", flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
