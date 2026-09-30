"""Bounded physics/observation checks for V1, without a learned policy."""

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for package in ("agent_world", "agent_tasks", "agent_rl"):
    sys.path.insert(0, str(ROOT / "source" / package))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Robotics-Deformable-Suspension-Flat-History-Transformer-v1")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--b0", type=float, default=None)
parser.add_argument("--w0", type=float, default=None)
parser.add_argument("--applied-feedback", action="store_true")
parser.add_argument("--standing-only", action="store_true")
parser.add_argument("--diagnose", action="store_true", help="Disable safety resets to inspect controller failure.")
parser.add_argument("--strict", action="store_true", help="Assert settled flat-ground speed and clearance (use >=600 steps).")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
launcher = AppLauncher(args)

import gymnasium as gym
import torch
import agent_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import quat_apply_inverse
from agent_tasks.direct.deformable_suspension import cfg_utils as du


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=4)
    cfg.external_cmd_override = True
    cfg.commands_world_frame = True
    cfg.episode_length_s = 30.0
    cfg.boundary_reset_enabled = False
    cfg.seed = 42
    if args.b0 is not None:
        cfg.adrc_b0 = args.b0
    if args.w0 is not None:
        cfg.adrc_eso_w0 = args.w0
    cfg.adrc_feedback_applied_torque = args.applied_feedback
    if args.diagnose:
        cfg.base_contact_death_after_iterations = 1000000000
        cfg.terminate_chassis_clearance = -1.0
        cfg.height_termination_margin = 1.0
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    actions = torch.zeros(4, 4, device=u.device)
    try:
        commands = ((0, 0, 0),) if args.standing_only else (
            (0, 0, 0), (1, 0, 0), (0, 0, 2 * math.pi), (0, 0, -2 * math.pi), (0.5, 0, 2 * math.pi))
        substeps = []
        original_apply = u._apply_action
        def record_apply():
            original_apply()
            substeps.append(torch.stack((
                u.robot.data.joint_pos[:, u._legs_idx].mean(),
                u.robot.data.joint_vel[:, u._legs_idx].abs().max(),
                u._leg_adrc.last_u.abs().max(),
                u._leg_adrc.applied_u.abs().max(),
                u.wheel_normal_forces.min(),
                (u.wheel_normal_forces > 3).all(-1).float().mean(),
                (u._leg_adrc.applied_u.abs() >= 24.9).float().mean(),
            )).detach())
        u._apply_action = record_apply
        for command in commands:
            substeps.clear()
            obs, _ = env.reset()
            assert obs["policy"].shape == (4, 256) and obs["critic"].shape == (4, 40)
            repeat = u._get_observations()["policy"]
            torch.testing.assert_close(obs["policy"], repeat)
            torch.testing.assert_close(u._history[:, 0], u._history[:, -1])
            acc = []
            resets = 0
            for step in range(args.steps):
                u.cmd_buf[:] = torch.tensor(command, device=u.device)
                obs, reward, terminated, timeout, _ = env.step(actions)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                resets += int((terminated | timeout).sum())
                if step > args.steps // 2:
                    data = u.robot.data
                    centers, _, _, _, _ = u._wheel_geometry_w()
                    centers_b = quat_apply_inverse(data.root_link_quat_w[:, None].expand(-1, 4, -1),
                                                   centers - data.root_link_pos_w[:, None])
                    predicted, _ = du.wheel_geometry(data.joint_pos[:, u._legs_idx])
                    geometry_error = (centers_b - predicted).abs().max()
                    acc.append(torch.stack((data.root_lin_vel_w[:, 0].mean(),
                                            data.root_lin_vel_w[:, 1].mean(),
                                            data.root_ang_vel_b[:, 2].mean(),
                                            (u.wheel_normal_forces > 3).all(-1).float().mean(),
                                            u.body_top_height.mean(), u.chassis_clearance.min(),
                                            geometry_error)))
            values = torch.stack(acc).mean(0).tolist()
            micro = torch.stack(substeps[len(substeps)//2:])
            print(f"SUBSTEP mean={micro.mean(0).tolist()} std={micro.std(0).tolist()} "
                  f"q={u.robot.data.joint_pos[:, u._legs_idx].mean(0).tolist()}", flush=True)
            print(f"PROBE cmd={command} vx/vy/wz={values[:3]} all_contact={values[3]:.3f} "
                  f"height={values[4]:.4f} clearance_min={values[5]:.4f} "
                  f"FK_error={values[6]:.6f} resets={resets}", flush=True)
            assert values[6] < 0.005, "URDF geometry/axis convention mismatch"
            if args.strict:
                assert not u._periodic and args.steps >= 600 and not args.diagnose
                assert resets == 0
                assert abs(values[0] - command[0]) < 0.2 and abs(values[1] - command[1]) < 0.1
                assert abs(values[2] - command[2]) < 0.2
                assert values[3] > 0.99 and values[4] <= 0.255 and 0.005 <= values[5] <= 0.010

        # Fresh high teleport + sensor reset must not leave a previous contact wrench.
        env.reset()
        ids = torch.arange(4, device=u.device)
        state = u.robot.data.root_state_w.clone()
        state[:, 2] += 0.5
        state[:, 7:] = 0.0
        u.robot.write_root_state_to_sim(state)
        u.contact_sensor.reset(ids)
        u.cmd_buf[:, 2] = 2 * math.pi
        u._apply_action()
        assert u._last_servo_force.abs().max() == 0
        print("PROBE airborne traction=0; history idempotence/reset and finite observations passed", flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    launcher.app.close()
    raise
else:
    launcher.app.close()
