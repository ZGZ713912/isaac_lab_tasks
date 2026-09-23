# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# =============================================================================

# Wheel_leg_V2 分档诊断：加载 checkpoint，对一组固定指令分别跑稳态，
# 输出跟踪误差、俯仰/横滚、力矩、摔倒率，用来定位“平地移动”问题。
#
# 用法（仓库根目录；headless）：
#   python scripts/rsl_rl/diagnose_wheel_leg_v2.py \
#       --task=Robotics-Wheel-Leg-V2-Flat-Play-v0 \
#       --checkpoint=logs/rsl_rl/wheel_leg_v2_flat_direct/2026-09-19_22-56-50/model_9999.pt \
#       --num_envs=32 --warmup=100 --steps=400 --headless --device=cuda:0
# =============================================================================

"""Launch Isaac Sim Simulator first."""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
for _pkg in ("agent_world", "agent_tasks", "agent_rl"):
    _pkg_root = os.path.join(_REPO_ROOT, "source", _pkg)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
_RSL_RL_SCRIPTS = os.path.join(_REPO_ROOT, "scripts", "rsl_rl")
if _RSL_RL_SCRIPTS not in sys.path:
    sys.path.insert(0, _RSL_RL_SCRIPTS)

import argparse
import csv
import math
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wheel_leg_V2 command-binned diagnostic.")
parser.add_argument("--task", type=str, default="Robotics-Wheel-Leg-V2-Flat-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXXX.pt")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--warmup", type=int, default=100, help="Steps discarded before measuring each command.")
parser.add_argument("--steps", type=int, default=400, help="Measured steps per command (100 Hz).")
parser.add_argument("--csv", type=str, default=None, help="CSV output path (default logs/debug/diag_<ts>.csv)")
parser.add_argument("--mode", type=str, default="bins", choices=("bins", "random", "drift"),
                    help="bins = fixed command list; random = contract-sampled commands; "
                         "drift = long straight-line tracking of path/heading.")
parser.add_argument("--switch", type=int, default=0,
                    help="bins mode: resample a random command every N measured steps (0 = hold fixed).")
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
import cli_args as rsl_cli_args  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

JOINTS = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]

# (label, vx, wz, height)
BINS = [
    ("stand_zero", 0.0, 0.0, 0.22),
    ("fwd_0.3", 0.3, 0.0, 0.22),
    ("fwd_0.6", 0.6, 0.0, 0.22),
    ("fwd_low", 0.4, 0.0, 0.22),
    ("fwd_mid", 1.0, 0.0, 0.22),
    ("fwd_high", 1.8, 0.0, 0.22),
    ("fwd_max", 2.0, 0.0, 0.22),
    ("back_low", -0.4, 0.0, 0.22),
    ("back_high", -1.5, 0.0, 0.22),
    ("back_max", -2.0, 0.0, 0.22),
    ("spin_pos", 0.0, 1.5, 0.22),
    ("spin_neg", 0.0, -1.5, 0.22),
    ("spin_max", 0.0, 2.0, 0.22),
    ("spin_2pi", 0.0, 2.0 * math.pi, 0.22),
    ("spin_3pi", 0.0, 3.0 * math.pi, 0.22),
    ("spin_4pi", 0.0, 4.0 * math.pi, 0.22),
    ("spin_4.5pi", 0.0, 4.5 * math.pi, 0.22),
    ("arc_fwd", 1.0, 0.8, 0.22),
    ("arc_back", -1.0, -0.8, 0.22),
    ("corner_fwd_spin", 2.0, 2.0, 0.22),
    ("corner_fwd_cspin", 2.0, -2.0, 0.22),
    ("corner_back_spin", -2.0, 2.0, 0.22),
    ("corner_back_cspin", -2.0, -2.0, 0.22),
    ("low_height_fwd", 1.0, 0.0, 0.20),
    ("high_height_fwd", 1.0, 0.0, 0.25),
]


def _mean(x: torch.Tensor) -> float:
    return float(torch.nan_to_num(x, nan=0.0).mean().item())


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    ns = argparse.Namespace(
        task=args_cli.task, device=args_cli.device, seed=None, run_name=None, logger=None,
        log_project_name=None, clip_actions=None, cmoe_router_temperature=None,
        moe_load_balancing_coef=None, cmoe_aux=None, experiment_name=None, resume=None,
        load_run=None, checkpoint=None,
    )
    agent_cfg = rsl_cli_args.parse_rsl_rl_cfg(args_cli.task, ns)
    agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    robot = unwrapped.robot
    torque_cap = unwrapped._torque_cap
    n = args_cli.num_envs
    rows = []

    def sample_random_commands() -> None:
        d = torch.rand(n, 3, device=env.device)
        unwrapped.commands[:, 0] = d[:, 0] * 4.0 - 2.0
        unwrapped.commands[:, 1] = d[:, 1] * 4.0 - 2.0
        unwrapped.commands[:, 2] = 0.20 + d[:, 2] * 0.05
        unwrapped._command_ticks_left[:] = unwrapped._command_period_ticks
        unwrapped._commands_due[:] = False

    print(f"[INFO] mode={args_cli.mode} {n} envs, warmup={args_cli.warmup}, "
          f"measure={args_cli.steps} (100 Hz), switch={args_cli.switch}")
    with torch.inference_mode():
        if args_cli.mode == "drift":
            from isaaclab.utils.math import euler_xyz_from_quat
            drift_bins = [
                ("stand", 0.0, 0.0, 0.22),
                ("fwd_1.0", 1.0, 0.0, 0.22),
                ("back_1.0", -1.0, 0.0, 0.22),
                ("spin_1.0", 0.0, 1.0, 0.22),
                ("arc_1.0_0.6", 1.0, 0.6, 0.22),
            ]
            for label, vx, wz, h in drift_bins:
                unwrapped._evaluation_command = (float(vx), float(wz), float(h))
                obs, _ = env.reset()
                p0 = robot.data.root_pos_w[:, :2].clone()
                q0 = robot.data.root_quat_w.clone()
                _, _, yaw0 = euler_xyz_from_quat(q0)
                rolls, pitches = [], []
                for _ in range(args_cli.steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    g = robot.data.projected_gravity_b
                    rolls.append(float(g[:, 1].mean()))
                    pitches.append(float(g[:, 0].mean()))
                p1 = robot.data.root_pos_w[:, :2]
                _, _, yaw1 = euler_xyz_from_quat(robot.data.root_quat_w)
                d = p1 - p0
                cy, sy = torch.cos(yaw0), torch.sin(yaw0)
                dx_b = cy * d[:, 0] + sy * d[:, 1]
                dy_b = -sy * d[:, 0] + cy * d[:, 1]
                dyaw = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))
                dist = dx_b.abs().clamp_min(1e-6)
                row = {"bin": label, "cmd_vx": vx, "cmd_wz": wz, "cmd_h": h,
                       "fwd_m": float(dx_b.mean()), "lat_m": float(dy_b.mean()),
                       "lat_abs_m": float(dy_b.abs().mean()), "yaw_deg": float(torch.rad2deg(dyaw).mean()),
                       "lat_per_m": float((dy_b.abs() / dist).mean()),
                       "roll_deg": float(torch.rad2deg(torch.asin(torch.clamp(torch.tensor(sum(rolls) / len(rolls)), -1, 1)))),
                       "pitch_deg": float(torch.rad2deg(torch.asin(torch.clamp(torch.tensor(sum(pitches) / len(pitches)), -1, 1)))),
                       "fall_rate": 0.0, "count": args_cli.steps * n, "vx_err": None, "mean_vx": None,
                       "mean_wz": None, "wz_err": None, "mean_h": None, "h_err": None,
                       "mean_speed": None, "tau_leg_max": None, "tau_wheel_max": None,
                       "wheel_speed": None, "wheel_sat": None}
                rows.append(row)
                print(f"  {label:12s} fwd={row['fwd_m']:+.2f}m lat_abs={row['lat_abs_m']:.2f}m "
                      f"yaw={row['yaw_deg']:+.1f}deg lat/m={row['lat_per_m']:.3f} "
                      f"roll={row['roll_deg']:+.1f} pitch={row['pitch_deg']:+.1f}")
        elif args_cli.mode == "random":
            unwrapped._evaluation_command = None
            obs, _ = env.reset()
            # per |cmd_vx| speed bin accumulators: [0,0.5),[0.5,1),[1,1.5),[1.5,2]
            edges = [0.0, 0.5, 1.0, 1.5, 2.01]
            s_acc = [{"vx": 0.0, "vx_err": 0.0, "wz_err": 0.0, "cnt": 0,
                      "roll": 0.0, "pitch": 0.0} for _ in range(len(edges) - 1)]
            falls = 0
            total = 0
            for _ in range(args_cli.steps):
                obs, _, dones, _ = env.step(policy(obs))
                done = dones.bool()
                falls += int(done.sum().item())
                total += n
                valid = ~done
                if not valid.any():
                    continue
                data = robot.data
                v = data.root_lin_vel_b[valid]
                w = data.root_ang_vel_b[valid]
                g = data.projected_gravity_b[valid]
                cmd = unwrapped.commands[valid]
                absvx = v[:, 0].abs()
                for k in range(len(edges) - 1):
                    m = (absvx >= edges[k]) & (absvx < edges[k + 1])
                    if not m.any():
                        continue
                    s_acc[k]["vx"] += float(v[m, 0].sum())
                    s_acc[k]["vx_err"] += float((v[m, 0] - cmd[m, 0]).abs().sum())
                    s_acc[k]["wz_err"] += float((w[m, 2] - cmd[m, 1]).abs().sum())
                    s_acc[k]["roll"] += float(g[m, 1].sum())
                    s_acc[k]["pitch"] += float(g[m, 0].sum())
                    s_acc[k]["cnt"] += int(m.sum())
            print(f"  overall fall_rate={falls/total*100:.1f}% over {total} env-steps")
            for k in range(len(edges) - 1):
                c = s_acc[k]["cnt"]
                if c == 0:
                    continue
                row = {
                    "bin": f"speed[{edges[k]:.1f},{edges[k+1]:.1f})", "cmd_vx": None,
                    "cmd_wz": None, "cmd_h": None,
                    "mean_vx": s_acc[k]["vx"] / c, "vx_err": s_acc[k]["vx_err"] / c,
                    "mean_wz": None, "wz_err": s_acc[k]["wz_err"] / c,
                    "mean_h": None, "h_err": None,
                    "pitch_deg": float(torch.rad2deg(torch.asin(torch.tensor(s_acc[k]["pitch"] / c))).item()),
                    "roll_deg": float(torch.rad2deg(torch.asin(torch.tensor(min(1.0, s_acc[k]["roll"] / c))))),
                    "mean_speed": None, "tau_leg_max": None, "tau_wheel_max": None,
                    "wheel_speed": None, "fall_rate": falls / total,
                    "wheel_sat": None, "count": c,
                }
                rows.append(row)
                print(f"  {row['bin']:18s} n={c:6d} vx={row['mean_vx']:+.3f} "
                      f"|vx_err|={row['vx_err']:.3f} |wz_err|={row['wz_err']:.3f} "
                      f"roll={row['roll_deg']:+.1f}deg pitch={row['pitch_deg']:+.1f}deg")
        else:
            for label, vx, wz, h in BINS:
                unwrapped._evaluation_command = (float(vx), float(wz), float(h))
                obs, _ = env.reset()
                for _ in range(max(args_cli.warmup, 0)):
                    obs, _, _, _ = env.step(policy(obs))

                acc = {k: torch.zeros(n, device=env.device) for k in
                       ("vx", "wz", "h", "gx", "gy", "speed", "tau_leg", "tau_wheel", "ewheel",
                        "wt_l", "wt_r", "slip", "aw_l", "aw_r")}
                sat = torch.zeros(n, device=env.device)
                falls = 0
                total = 0
                for t in range(args_cli.steps):
                    if args_cli.switch > 0 and t > 0 and t % args_cli.switch == 0:
                        unwrapped._evaluation_command = None
                        sample_random_commands()
                    obs, _, dones, _ = env.step(policy(obs))
                    done = dones.bool()
                    falls += int(done.sum().item())
                    total += n
                    valid = ~done
                    if not valid.any():
                        continue
                    data = robot.data
                    v = data.root_lin_vel_b[valid]
                    w = data.root_ang_vel_b[valid]
                    g = data.projected_gravity_b[valid]
                    tau = unwrapped.torques[valid]
                    jv = data.joint_vel[valid][:, unwrapped._joint_ids]
                    acc["vx"][valid] += v[:, 0]
                    acc["wz"][valid] += w[:, 2]
                    acc["h"][valid] += unwrapped._base_height()[valid]
                    acc["gx"][valid] += g[:, 0]
                    acc["gy"][valid] += g[:, 1]
                    acc["speed"][valid] += v[:, :2].norm(dim=-1)
                    acc["tau_leg"][valid] += tau[:, unwrapped._leg_ids].abs().amax(dim=-1)
                    acc["tau_wheel"][valid] += tau[:, unwrapped._wheel_ids].abs().amax(dim=-1)
                    acc["ewheel"][valid] += jv[:, unwrapped._wheel_ids].abs().amax(dim=-1)
                    # 轮控制诊断：解码后轮目标 / 原始轮动作 / 轮底滑移（判断烧胎）
                    acc["wt_l"][valid] += unwrapped.wheel_targets[:, 0]
                    acc["wt_r"][valid] += unwrapped.wheel_targets[:, 1]
                    acc["aw_l"][valid] += unwrapped.actions[:, unwrapped._wheel_ids[0]]
                    acc["aw_r"][valid] += unwrapped.actions[:, unwrapped._wheel_ids[1]]
                    acc["slip"][valid] += unwrapped._wheel_slip().amax(dim=-1)
                    sat[valid] += (tau[:, unwrapped._wheel_ids].abs()
                                   >= 0.99 * torque_cap[unwrapped._wheel_ids]).any(dim=-1).float()

                denom = max(args_cli.steps, 1)
                row = {
                    "bin": label, "cmd_vx": vx, "cmd_wz": wz, "cmd_h": h,
                    "mean_vx": _mean(acc["vx"] / denom),
                    "vx_err": abs(_mean(acc["vx"] / denom) - vx),
                    "mean_wz": _mean(acc["wz"] / denom),
                    "wz_err": abs(_mean(acc["wz"] / denom) - wz),
                    "mean_h": _mean(acc["h"] / denom),
                    "h_err": abs(_mean(acc["h"] / denom) - h),
                    "pitch_deg": _mean(torch.rad2deg(torch.asin((-acc["gx"] / denom).clamp(-1, 1)))),
                    "roll_deg": _mean(torch.rad2deg(torch.asin((acc["gy"] / denom).clamp(-1, 1)))),
                    "mean_speed": _mean(acc["speed"] / denom),
                    "tau_leg_max": _mean(acc["tau_leg"] / denom),
                    "tau_wheel_max": _mean(acc["tau_wheel"] / denom),
                    "wheel_speed": _mean(acc["ewheel"] / denom),
                    "wheel_tgt_l": _mean(acc["wt_l"] / denom),
                    "wheel_tgt_r": _mean(acc["wt_r"] / denom),
                    "wheel_act_l": _mean(acc["aw_l"] / denom),
                    "wheel_act_r": _mean(acc["aw_r"] / denom),
                    "wheel_slip_max": _mean(acc["slip"] / denom),
                    "fall_rate": falls / total,
                    "wheel_sat": _mean(sat / denom),
                    "count": total,
                }
                rows.append(row)
                print(f"  {label:16s} cmd(vx={vx:+.2f},wz={wz:+.2f},h={h:.2f}) -> "
                      f"vx={row['mean_vx']:+.3f}(err {row['vx_err']:.3f}) "
                      f"wz={row['mean_wz']:+.3f}(err {row['wz_err']:.3f}) "
                      f"h={row['mean_h']:.3f}(err {row['h_err']:.3f}) "
                      f"pitch={row['pitch_deg']:+.1f} roll={row['roll_deg']:+.1f} "
                      f"tau_leg={row['tau_leg_max']:.1f} tau_wheel={row['tau_wheel_max']:.2f} "
                      f"wtgt=({row['wheel_tgt_l']:+.1f},{row['wheel_tgt_r']:+.1f}) "
                      f"slip={row['wheel_slip_max']:.3f} "
                      f"sat={row['wheel_sat']*100:.1f}% fall={row['fall_rate']*100:.1f}%")

    env.close()

    # ---- summary + CSV ----
    print("\n" + "=" * 100)
    nz = [r for r in rows if r["cmd_vx"] != 0.0 or r["cmd_wz"] != 0.0]
    if nz:
        import statistics
        print(f"  moving bins: mean|vx_err|={statistics.mean(r['vx_err'] for r in nz):.3f} m/s, "
              f"mean|wz_err|={statistics.mean(r['wz_err'] for r in nz):.3f} rad/s, "
              f"mean fall={statistics.mean(r['fall_rate'] for r in nz)*100:.1f}%, "
              f"mean wheel_sat={statistics.mean(r['wheel_sat'] for r in nz)*100:.1f}%")

    csv_path = args_cli.csv or os.path.join(
        _REPO_ROOT, "logs", "debug", f"diag_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV -> {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
