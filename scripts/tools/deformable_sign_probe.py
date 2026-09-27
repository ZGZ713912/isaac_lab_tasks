# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Deformable 主动悬挂 —— 阶段 0 仿真探针（只读验证，不参与训练）
#
# 目的：
#   1) 在真实物理里确认“腿动作 -> 车身倾斜”的符号，从而定 tilt_leg_q_sign；
#   2) 实测低车身下 q 的安全上限（底盘触地前能压低多少）。
#
# 用法（仓库根目录，headless）：
#   python scripts/tools/deformable_sign_probe.py --device=cuda:0
# =============================================================================
"""Phase-0 sim probe for the deformable active-suspension tilt correction sign."""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _pkg in ("agent_world", "agent_tasks", "agent_rl"):
    _pkg_root = os.path.join(_REPO_ROOT, "source", _pkg)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deformable tilt-sign / travel probe.")
parser.add_argument("--task", type=str, default="Robotics-Deformable-Suspension-Play-v0")
parser.add_argument("--settle", type=int, default=250, help="Settle steps before measuring.")
parser.add_argument("--measure", type=int, default=120, help="Steps averaged per configuration.")
parser.add_argument("--delta", type=float, default=0.15, help="Leg q offset used for tilt configs (rad).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import agent_world  # noqa: F401,E402
import agent_tasks  # noqa: F401,E402
from agent_tasks.direct.deformable_suspension import cfg_utils as du  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True


def _steer(env, q_des: torch.Tensor) -> torch.Tensor:
    """把期望腿角（4 维，按 ORDERED_LEG_JOINT_NAMES 顺序）转成 action。"""
    u = env.unwrapped
    q_cmd = u.q_cmd.unsqueeze(-1)  # (N,1)
    scale = u.cfg.leg_action_scale
    action = (q_des - q_cmd) / scale
    return action.clamp(-10.0, 10.0)


def _run(env, q_des: torch.Tensor, settle: int, measure: int) -> dict:
    """固定腿角命令跑 settle+measure 步，返回末段均值。"""
    u = env.unwrapped
    act = _steer(env, q_des)
    buf = {"pgb": [], "height": [], "q": [], "base_force": [], "wheel_force": []}
    total = int(settle) + int(measure)
    for i in range(total):
        env.step(act)
        if i >= int(settle):
            buf["pgb"].append(u.robot.data.projected_gravity_b.clone())
            buf["height"].append(u.base_height.clone())
            buf["q"].append(u.robot.data.joint_pos[:, u._legs_idx].clone())
            buf["base_force"].append(
                torch.norm(
                    u.contact_sensor.data.net_forces_w[:, u._base_contact_idx, :], dim=-1
                ).max(dim=-1).values.clone()
            )
            buf["wheel_force"].append(u.wheel_contact_forces.clone())
    out = {}
    for k, v in buf.items():
        t = torch.stack(v, dim=0).mean(dim=0)[0]
        out[k] = t
    return out


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.play = True
    env_cfg.events = None
    env_cfg.enable_chassis_servo = False
    env_cfg.episode_length_s = 1.0e6
    env_cfg.boundary_reset_enabled = False
    env_cfg.termination_roll_deg = 90.0
    env_cfg.termination_pitch_deg = 90.0
    env_cfg.terminate_base_height_low = -1.0e9
    env_cfg.base_contact_death_after_iterations = 1_000_000_000
    env_cfg.leg_target_upper_limit = du.LEG_UPPER_LIMIT  # probe 放开，观察真实物理
    env_cfg.seed = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()

    names = list(du.ORDERED_LEG_JOINT_NAMES)
    print("\n================ Phase-0 probe ================")
    print(f"leg joint order: {names}   (leg1=front-right, leg2=front-left, leg3=rear-left, leg4=rear-right)")
    print(f"Q_LOW={du.Q_LOW:.4f}  LEG_UPPER_LIMIT={du.LEG_UPPER_LIMIT:.4f}  BODY_BOTTOM_OFFSET={du.BODY_BOTTOM_OFFSET:.4f}")

    # ---- 1) 符号测试：不同腿角构型产生的车身倾角 ----
    print("\n---- [1] leg pattern -> body tilt sign ----")
    d = float(args_cli.delta)
    base = torch.full((1, 4), float(du.Q_LOW), device=u.device)
    configs = {
        "all_low":      [0.0, 0.0, 0.0, 0.0],
        "front_raise":  [-d, -d, 0.0, 0.0],   # leg1,2 减 q -> 抬前
        "rear_raise":   [0.0, 0.0, -d, -d],   # leg3,4 减 q -> 抬后
        "left_raise":   [0.0, -d, -d, 0.0],   # leg2,3 (y=+L) 抬
        "right_raise":  [-d, 0.0, 0.0, -d],   # leg1,4 (y=-L) 抬
    }
    results = {}
    for name, off in configs.items():
        q_des = (base + torch.tensor(off, device=u.device).unsqueeze(0)).clamp(
            du.LEG_LOWER_LIMIT, du.LEG_UPPER_LIMIT
        )
        r = _run(env, q_des, args_cli.settle, args_cli.measure)
        results[name] = r
        pgb = r["pgb"]
        tilt = float(torch.sqrt(torch.clamp(pgb[0] ** 2 + pgb[1] ** 2, min=0.0)))
        print(
            f"{name:12s} pgb=({pgb[0]:+.4f},{pgb[1]:+.4f},{pgb[2]:+.4f}) "
            f"|tilt|={tilt:.4f} ({tilt*57.2958:5.2f}deg) h={float(r['height']):.4f} "
            f"q={[round(float(x),3) for x in r['q']]}"
        )

    print("\n  sign interpretation:")
    b = results["all_low"]["pgb"]
    for name in ("front_raise", "rear_raise", "left_raise", "right_raise"):
        p = results[name]["pgb"]
        print(
            f"    {name:12s} delta_pgb_x={float(p[0]-b[0]):+.4f}  delta_pgb_y={float(p[1]-b[1]):+.4f}"
        )

    # ---- 2) 行程测试：统一压低 q，看底盘何时触地 ----
    print("\n---- [2] q upper travel vs chassis clearance ----")
    print(f"{'q':>8} {'base_h':>8} {'clearance':>10} {'baseF(max)':>10} {'wheelF(min)':>12}")
    travel = {}
    for delta in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20, 0.30):
        q = float(du.Q_LOW) + delta
        q_des = torch.full((1, 4), q, device=u.device)
        r = _run(env, q_des, max(60, args_cli.settle // 3), args_cli.measure)
        h = float(r["height"])
        clearance = h + du.BODY_BOTTOM_OFFSET
        base_f = float(r["base_force"])
        wheel_min = float(r["wheel_force"].min())
        travel[q] = clearance
        print(f"{q:8.4f} {h:8.4f} {clearance:10.4f} {base_f:10.3f} {wheel_min:12.3f}")

    # ---- 3) 闭环验证：用环境自身的 _get_tilt_leg_targets 目标，看倾斜是否下降 ----
    print("\n---- [3] closed-loop: does env target reduce tilt? ----")
    closed = {}
    for start in ("front_raise", "rear_raise", "left_raise", "right_raise"):
        # 先进入该构型
        off = configs[start]
        q_des = (base + torch.tensor(off, device=u.device).unsqueeze(0)).clamp(
            du.LEG_LOWER_LIMIT, du.LEG_UPPER_LIMIT
        )
        r0 = _run(env, q_des, max(60, args_cli.settle // 3), args_cli.measure)
        pgb0 = r0["pgb"]
        tilt0 = float(torch.sqrt(torch.clamp(pgb0[0] ** 2 + pgb0[1] ** 2, min=0.0)))
        # 用环境自己的公式求目标，再闭环跑到目标
        tgt, qd = u._get_tilt_leg_targets(pgb0.unsqueeze(0))
        r1 = _run(env, tgt, args_cli.settle, args_cli.measure)
        pgb1 = r1["pgb"]
        tilt1 = float(torch.sqrt(torch.clamp(pgb1[0] ** 2 + pgb1[1] ** 2, min=0.0)))
        closed[start] = (tilt0, tilt1)
        print(
            f"{start:12s} q_delta={[round(float(x),3) for x in qd[0]]}  "
            f"tilt {tilt0*57.2958:5.2f}deg -> {tilt1*57.2958:5.2f}deg  "
            f"({'OK: reduced' if tilt1 < tilt0 else 'BAD: increased'})"
        )

    print("\n================ conclusions ================")
    fb = results["front_raise"]["pgb"]
    rb = results["rear_raise"]["pgb"]
    front_raises = "pgb_x decreases (nose up)" if float(fb[0] - b[0]) < 0 else "pgb_x increases (nose down)"
    print(f"  front_raise -> {front_raises}")
    print(f"  left_raise  -> delta_pgb_y={float(results['left_raise']['pgb'][1]-b[1]):+.4f}")
    # 期望：给定 pgb_x>0 (前低)，应抬前。若 front_raise 使 pgb_x 减小，则"抬前修前低"成立。
    if float(fb[0] - b[0]) < 0:
        print("  => to fix pgb_x>0 (nose-down) you must RAISE the front -> tilt_leg_q_sign should be -1.0")
    else:
        print("  => to fix pgb_x>0 (nose-down) you must RAISE the rear -> tilt_leg_q_sign stays +1.0")
    safe = [q for q, c in travel.items() if c > 0.005]
    print(f"  chassis clearance > 5mm for q <= {max(safe) if safe else float('nan'):.4f}")
    n_ok = sum(1 for t0, t1 in closed.values() if t1 < t0)
    print(f"  closed-loop tilt reduced in {n_ok}/{len(closed)} starting attitudes")
    print("=============================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
