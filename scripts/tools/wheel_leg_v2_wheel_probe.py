# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V2 轮系 probe（无策略，只读诊断，不参与训练）
#
# 目的：确认「策略轮动作符号 -> 车体运动」的真实映射，并实测轮速伺服/力矩能力。
#
# 背景（纯几何推导，base 系）：
#   L_joint3 转轴 = +Y，R_joint3 转轴 = -Y（腿关节全为横轴，故该方向在工作空间内恒定）。
#   直行要求两轮绕世界 -Y... 即 ω_y>0，对应 q̇_L>0 且 q̇_R<0  -> 左右关节速度【反号】
#   自转要求两轮世界旋转相反，对应 q̇_L>0 且 q̇_R>0  -> 左右关节速度【同号】
#   即 joint 动作空间里「同号=自转、反号=直行」，与直觉相反。
#
# 本脚本用 action=[0,0,wL,0,0,wR]（腿维全 0 -> 腿保持名义位形）直接驱动轮关节速度目标
# = action * wheel_velocity_scale，测量 base vx/wz 与轮世界角速度，直接得到符号映射。
#
# 用法（仓库根目录）：
#   # A1 固定 base：稳定，测轮速伺服/力矩/地面反力方向
#   ./run_gui.sh python scripts/tools/wheel_leg_v2_wheel_probe.py --fix-base --no-gas-spring
#   # A2 自由 base：真实落地，测 vx/wz 响应
#   ./run_gui.sh python scripts/tools/wheel_leg_v2_wheel_probe.py --no-gas-spring
#   # 气簧 A/B
#   ./run_gui.sh python scripts/tools/wheel_leg_v2_wheel_probe.py --no-gas-spring
#   ./run_gui.sh python scripts/tools/wheel_leg_v2_wheel_probe.py
# =============================================================================

"""Wheel_leg_V2 open-loop wheel actuator probe."""

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
import json
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wheel_leg_V2 wheel sign / authority probe.")
parser.add_argument("--task", type=str, default="Robotics-Wheel-Leg-V2-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--settle", type=int, default=150, help="Steps before measuring (100 Hz).")
parser.add_argument("--measure", type=int, default=150, help="Steps averaged per config (100 Hz).")
parser.add_argument("--fix-base", action="store_true", default=False,
                    help="A1: fix the articulation root (stable; measures servo + reaction force).")
parser.add_argument("--gas-spring", dest="gas_spring", action="store_true", default=None,
                    help="Force gas spring ON.")
parser.add_argument("--no-gas-spring", dest="gas_spring", action="store_false",
                    help="Force gas spring OFF (isolate the wheel chain).")
parser.add_argument("--json", type=str, default=None, help="JSON output path.")
parser.add_argument("--wheel-kd", type=float, default=None,
                    help="Override contract wheel actuator kd (velocity servo gain) for a sweep.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import agent_world  # noqa: F401,E402
import agent_tasks  # noqa: F401,E402
from agent_world.assets.wheel_leg_V2 import WheelLegV2_CFG  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# (label, wheel action L, wheel action R); action * wheel_velocity_scale = joint velocity target
# 语义（已按 wheel_joint_sign=[+1,-1] 修正）：正动作 = 车轮正向滚动 = 车体前进。
CONFIGS = [
    ("zero",       0.0,   0.0),
    ("fwd_1",     +1.0,  +1.0),
    ("fwd_3",     +3.0,  +3.0),
    ("fwd_6",     +6.0,  +6.0),
    ("fwd_10",   +10.0, +10.0),
    ("fwd_20",   +20.0, +20.0),
    ("back_3",    -3.0,  -3.0),
    ("yaw_pos",   +3.0,  -3.0),
    ("yaw_neg",   -3.0,  +3.0),
]


def _num(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).mean().item())


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e6
    if args_cli.gas_spring is not None:
        env_cfg.gas_spring_enabled = bool(args_cli.gas_spring)
    gas_spring = bool(env_cfg.gas_spring_enabled)

    # 固定/自由 base 必须在 env.__init__ 里 robot_cfg 构造之前改到资产 cfg 上。
    WheelLegV2_CFG.spawn.articulation_props.fix_root_link = bool(args_cli.fix_base)

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped

    # 关掉所有终止与看门狗，probe 期间不 reset。
    u.contract["termination"].update({
        "contact_force_threshold": 1.0e9,
        "max_tilt_deg": 89.9,
        "min_base_height": -1.0e9,
        "knee_limit_tolerance": 1.0e9,
    })
    u.cfg.closure_error_tolerance = 0.0

    if args_cli.wheel_kd is not None:
        u.contract["actuators"]["wheel"]["kd"] = float(args_cli.wheel_kd)

    robot = u.robot
    scale = float(u.contract["actions"]["wheel_velocity_scale"])
    wheel_ids = u._wheel_ids                      # 关节表下标 [L_joint3, R_joint3]
    wheel_bodies = u._wheel_robot_body_ids        # robot 刚体表下标
    wheel_contact = u._wheel_body_ids             # 接触传感器刚体下标
    cap = float(u.contract["actuators"]["wheel"]["effort_limit"])

    print("\n================ Wheel_leg_V2 wheel probe ================")
    print(f"task={args_cli.task}  envs={args_cli.num_envs}  fix_base={bool(args_cli.fix_base)}")
    print(f"gas_spring={gas_spring}  wheel_velocity_scale={scale}  joint_order={u.joint_names}")
    print(f"wheel_joint_sign={u.contract['actions'].get('wheel_joint_sign')}  "
          f"wheel_kd={u.contract['actuators']['wheel']['kd']}")
    print(f"wheel joint ids={wheel_ids.tolist()}  wheel body ids={wheel_bodies.tolist()}")
    print(f"nominal_base_height={u.contract['asset']['nominal_base_height']}  effort_cap={cap:.4f} N*m")

    rows = []
    for label, wl, wr in CONFIGS:
        try:
            env.reset()
            action = torch.zeros((u.num_envs, 6), device=u.device)
            action[:, wheel_ids[0]] = wl
            action[:, wheel_ids[1]] = wr

            acc = {k: torch.zeros(u.num_envs, device=u.device) for k in
                   ("tgt_l", "tgt_r", "jv_l", "jv_r", "tau_l", "tau_r",
                    "wvy_l", "wvy_r", "fx_l", "fx_r", "fz_l", "fz_r",
                    "vx", "vy", "wz", "pitch", "roll", "h", "slip", "wx")}
            term_frac = 0.0
            tout_frac = 0.0
            trace = []
            total = max(int(args_cli.settle), 0) + max(int(args_cli.measure), 1)
            for i in range(total):
                _, _, term, tout, _ = env.step(action)
                if i % 10 == 0:
                    wvy_all = robot.data.body_ang_vel_w[:, wheel_bodies][..., 1]
                    trace.append({
                        "i": i,
                        "vx": _num(robot.data.root_lin_vel_b[:, 0]),
                        "wz": _num(robot.data.root_ang_vel_b[:, 2]),
                        "wvy_l": _num(wvy_all[:, 0]),
                        "wvy_r": _num(wvy_all[:, 1]),
                    })
                if i < int(args_cli.settle):
                    continue
                term_frac = _num(term.float())
                tout_frac = _num(tout.float())
                data = robot.data
                # _wheel_ids 是合同动作序(0..5)下标；先按 _joint_ids 取 6 个驱动关节再选轮。
                jv = data.joint_vel[:, u._joint_ids][:, wheel_ids]
                tau = u.torques[:, wheel_ids]
                wvy = data.body_ang_vel_w[:, wheel_bodies][..., 1]
                force = u.contact_sensor.data.net_forces_w[:, wheel_contact, :]
                acc["tgt_l"] += u.wheel_targets[:, 0]
                acc["tgt_r"] += u.wheel_targets[:, 1]
                acc["jv_l"] += jv[:, 0]
                acc["jv_r"] += jv[:, 1]
                acc["tau_l"] += tau[:, 0]
                acc["tau_r"] += tau[:, 1]
                acc["wvy_l"] += wvy[:, 0]
                acc["wvy_r"] += wvy[:, 1]
                acc["fx_l"] += force[:, 0, 0]
                acc["fx_r"] += force[:, 1, 0]
                acc["fz_l"] += force[:, 0, 2]
                acc["fz_r"] += force[:, 1, 2]
                acc["vx"] += data.root_lin_vel_b[:, 0]
                acc["vy"] += data.root_lin_vel_b[:, 1]
                acc["wz"] += data.root_ang_vel_b[:, 2]
                acc["pitch"] += data.projected_gravity_b[:, 0]
                acc["roll"] += data.projected_gravity_b[:, 1]
                acc["h"] += u._base_height()
                acc["slip"] += u._wheel_slip().amax(dim=-1)
                acc["wx"] += u._wheel_x_relative_to_root().amin(dim=-1)
            n = max(int(args_cli.measure), 1)
            m = {k: _num(v / n) for k, v in acc.items()}
            row = {
                "bin": label, "act_l": wl, "act_r": wr,
                "target_l_rad_s": wl * scale, "target_r_rad_s": wr * scale,
                "jv_l": m["jv_l"], "jv_r": m["jv_r"],
                "wvy_l": m["wvy_l"], "wvy_r": m["wvy_r"],
                "tau_l": m["tau_l"], "tau_r": m["tau_r"],
                "tau_l_util": abs(m["tau_l"]) / cap, "tau_r_util": abs(m["tau_r"]) / cap,
                "force_x_l": m["fx_l"], "force_x_r": m["fx_r"],
                "force_z_l": m["fz_l"], "force_z_r": m["fz_r"],
                "vx": m["vx"], "vy": m["vy"], "wz": m["wz"],
                "pitch_deg": float(torch.rad2deg(torch.asin(torch.tensor(m["pitch"]).clamp(-1, 1)))),
                "roll_deg": float(torch.rad2deg(torch.asin(torch.tensor(m["roll"]).clamp(-1, 1)))),
                "height_m": m["h"], "wheel_slip_max": m["slip"], "wheel_x_rel": m["wx"],
                "terminated_frac": term_frac, "timeout_frac": tout_frac,
                "trace": trace,
            }
            rows.append(row)
            print(f"  {label:15s} act=({wl:+.1f},{wr:+.1f}) tgt=({wl*scale:+.0f},{wr*scale:+.0f}) "
                  f"jv=({row['jv_l']:+.2f},{row['jv_r']:+.2f}) wvy=({row['wvy_l']:+.2f},{row['wvy_r']:+.2f}) "
                  f"|tau|=({abs(row['tau_l']):.3f},{abs(row['tau_r']):.3f}) "
                  f"Fx=({row['force_x_l']:+.3f},{row['force_x_r']:+.3f}) "
                  f"vx={row['vx']:+.3f} wz={row['wz']:+.3f} "
                  f"pitch={row['pitch_deg']:+.1f} h={row['height_m']:.3f} "
                  f"slip={row['wheel_slip_max']:.3f} wx={row['wheel_x_rel']:+.3f}")
        except Exception as error:  # noqa: BLE001
            print(f"  {label:15s} FAILED: {type(error).__name__}: {error}")
            rows.append({"bin": label, "error": f"{type(error).__name__}: {error}"})

    print("\n  sign reading (free base): vx>0 => forward; wz>0 => left yaw")
    print("  sign reading (fixed base): force_x>0 => wheel pushes robot forward")
    for r in rows:
        if "vx" not in r:
            continue
        print(f"    {r['bin']:15s} vx={r['vx']:+.4f}  wz={r['wz']:+.4f}  "
              f"wvy_l={r['wvy_l']:+.3f} wvy_r={r['wvy_r']:+.3f}  "
              f"Fx_l={r['force_x_l']:+.2f} Fx_r={r['force_x_r']:+.2f}")

    out = args_cli.json or os.path.join(
        _REPO_ROOT, "logs", "debug",
        f"wheel_leg_v2_wheel_probe_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "task": args_cli.task, "num_envs": args_cli.num_envs,
        "fix_base": bool(args_cli.fix_base), "gas_spring": gas_spring,
        "wheel_velocity_scale": scale, "joint_names": list(u.joint_names),
        "wheel_joint_ids": wheel_ids.tolist(), "wheel_body_ids": wheel_bodies.tolist(),
        "effort_cap": cap, "settle": args_cli.settle, "measure": args_cli.measure,
        "results": rows,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  JSON -> {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
