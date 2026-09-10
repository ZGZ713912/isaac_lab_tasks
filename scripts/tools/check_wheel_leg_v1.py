# =============================================================================
# Wheel_leg_V1 资产 / 物理标定探针（不训练，只回答"这台机器人现在能不能站住"）
#
# 回答四个问题：
#   1) USD 装进来对不对：关节数/关节名/连杆数/质量惯量是否为真（不是空壳）
#   2) 零位姿态（q=0）落地后是什么样：base 高度、俯仰角、轮子是否着地
#   3) 给定关节姿态下能站多高：--pose 指定 L_joint1/L_joint2/R_joint1/R_joint2
#   4) 站立姿态搜索：--grid 扫一批 (q1, q2) 候选，按"活过多少步"排序
#
# 用法（仓库根目录、isaaclab 环境、必须 unset LD_LIBRARY_PATH）：
#   unset LD_LIBRARY_PATH && conda activate isaaclab
#   python scripts/tools/check_wheel_leg_v1.py --steps 300
#   python scripts/tools/check_wheel_leg_v1.py --pose L_joint1=-1.06,L_joint2=-1.80 \
#       --pose R_joint1=-1.06,R_joint2=-1.80
#   python scripts/tools/check_wheel_leg_v1.py --grid --steps 200
#
# 输出：stdout 表格 + JSON（--out，默认 logs/debug/wheel_leg_v1_probe.json）
# =============================================================================
"""Calibration probe for the Wheel_leg_V1 asset (no training, read-only)."""

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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wheel_leg_V1 asset / physics calibration probe.")
parser.add_argument("--task", type=str, default="Robotics-Wheel-Leg-V1-Flat-v0", help="Task name to probe.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (keep 1 for readable numbers).")
parser.add_argument("--steps", type=int, default=300, help="Sim steps to run per pose.")
parser.add_argument("--settle", type=int, default=10, help="Steps before the first measurement (let it land).")
parser.add_argument(
    "--pose",
    action="append",
    default=[],
    metavar="NAME=VALUE",
    help="Joint pose override, repeatable, e.g. --pose L_joint1=-1.06",
)
parser.add_argument("--grid", action="store_true", default=False, help="Scan a grid of (q1, q2) poses.")
parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible random resets.")
parser.add_argument("--out", type=str, default="logs/debug/wheel_leg_v1_probe.json", help="JSON output path.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import agent_tasks  # noqa: E402,F401  (triggers gym registration)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

WHEEL_HINT = "link3"  # 轮子连杆名 L_link3 / R_link3（此前误写 joint3，导致 wheel_centers 一直为空）


def _quat_to_pitch_deg(quat_w: torch.Tensor) -> torch.Tensor:
    """Return pitch (rotation about body Y) in degrees from a wxyz quaternion."""
    w, x, y, z = quat_w.unbind(-1)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    return torch.rad2deg(torch.asin(sinp))


def _quat_to_roll_deg(quat_w: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_w.unbind(-1)
    sinr = torch.clamp(2.0 * (w * x + y * z), -1.0, 1.0)
    return torch.rad2deg(torch.asin(sinr))


def build_pose_overrides(pose_args: list) -> dict:
    if not pose_args:
        return None
    pose = {}
    for item in pose_args:
        if "=" not in item:
            raise SystemExit(f"--pose expects NAME=VALUE, got: {item}")
        name, value = item.split("=", 1)
        pose[name.strip()] = float(value)
    return pose


def describe_asset(env) -> dict:
    """Static description of what the USD actually brought in."""
    robot = env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    masses = robot.root_physx_view.get_masses()[0].tolist()
    inertias = robot.root_physx_view.get_inertias()[0]
    inertia_norm = [round(float(inertias[i].norm()), 9) for i in range(inertias.shape[0])]
    return {
        "num_joints": len(joint_names),
        "num_bodies": len(body_names),
        "joint_names": joint_names,
        "body_names": body_names,
        "body_masses": [round(float(m), 5) for m in masses],
        "body_inertia_norms": inertia_norm,
        "total_mass": round(float(sum(masses)), 4),
        "zero_inertia_bodies": [body_names[i] for i, n in enumerate(inertia_norm) if n < 1e-9],
        "zero_mass_bodies": [body_names[i] for i, m in enumerate(masses) if float(m) < 1e-9],
        "joint_limits_lower": [round(float(v), 4) for v in robot.data.joint_limits[0][:, 0].tolist()],
        "joint_limits_upper": [round(float(v), 4) for v in robot.data.joint_limits[0][:, 1].tolist()],
    }


def sample_state(env) -> dict:
    robot = env.scene["robot"]
    root_pos_w = robot.data.root_pos_w[0]
    root_quat_w = robot.data.root_quat_w[0]
    body_names = list(robot.data.body_names)
    body_pos_w = robot.data.body_pos_w[0]

    wheel_rows = [i for i, n in enumerate(body_names) if WHEEL_HINT in n]
    wheels = {
        body_names[i]: {
            "x": round(float(body_pos_w[i, 0]), 4),
            "z": round(float(body_pos_w[i, 2]), 4),
        }
        for i in wheel_rows
    }

    state = {
        "base_z": round(float(root_pos_w[2]), 4),
        "base_pitch_deg": round(float(_quat_to_pitch_deg(root_quat_w.unsqueeze(0))[0]), 2),
        "base_roll_deg": round(float(_quat_to_roll_deg(root_quat_w.unsqueeze(0))[0]), 2),
        "base_lin_vel": [round(float(v), 3) for v in robot.data.root_lin_vel_b[0].tolist()],
        "base_ang_vel": [round(float(v), 3) for v in robot.data.root_ang_vel_b[0].tolist()],
        "wheel_centers": wheels,
        "base_z_minus_wheel_z": (
            round(float(root_pos_w[2] - body_pos_w[wheel_rows[0], 2]), 4) if wheel_rows else float("nan")
        ),
        "joint_pos": {n: round(float(robot.data.joint_pos[0, i]), 4) for i, n in enumerate(robot.data.joint_names)},
        "joint_vel": {n: round(float(robot.data.joint_vel[0, i]), 4) for i, n in enumerate(robot.data.joint_names)},
    }
    sensors = getattr(env.scene, "sensors", None) or {}
    contact_sensor = sensors.get("contact_forces") if hasattr(sensors, "get") else None
    if contact_sensor is not None:
        forces = contact_sensor.data.net_forces_w_history
        if forces is not None and forces.numel() > 0:
            peak = forces[0].norm(dim=-1).max(dim=0).values
            state["contact_force_peak_per_body"] = {
                n: round(float(peak[i]), 3) for i, n in enumerate(contact_sensor.body_names)
            }
    return state


def measure(env, pose, steps: int, settle: int) -> dict:
    """Roll out the env with zero policy action under the given pose; report what happened."""
    robot = env.scene["robot"]
    env.reset()
    if pose:
        names = list(pose.keys())
        joint_ids, _ = robot.find_joints(names)
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel)
        for i, jid in enumerate(joint_ids):
            joint_pos[:, jid] = pose[names[i]]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        env.scene.write_data_to_sim()

    action = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
    samples = []
    terminated_at = None
    for step in range(steps):
        _obs, _rew, terminated, truncated, _info = env.step(action)
        if bool(terminated.any()) or bool(truncated.any()):
            terminated_at = step
            break
        if step >= settle and step % 10 == 0:
            samples.append(sample_state(env))

    final = sample_state(env)
    start = samples[0] if samples else final
    return {
        "pose": pose,
        "steps_survived": steps if terminated_at is None else terminated_at,
        "survived": terminated_at is None,
        "start": start,
        "final": final,
        "trajectory_head": samples[:5],
    }


def grid_candidates() -> list:
    """(q1, q2) candidates around the sagittal solution that puts the wheel under the hip."""
    from itertools import product

    return [
        {"L_joint1": q1, "L_joint2": q2, "R_joint1": q1, "R_joint2": q2}
        for q1, q2 in product([-1.20, -1.06, -0.90, -0.80, 0.0], [-1.90, -1.80, -1.70, -0.60, 0.0])
    ]


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    torch.manual_seed(args_cli.seed)

    print("\n================ Wheel_leg_V1 资产自检 ================")
    asset = describe_asset(env)
    for key in (
        "num_joints",
        "num_bodies",
        "total_mass",
        "joint_names",
        "body_names",
        "body_masses",
        "zero_mass_bodies",
        "zero_inertia_bodies",
        "joint_limits_lower",
        "joint_limits_upper",
    ):
        print(f"  {key}: {asset[key]}")

    results = []
    if args_cli.grid:
        print("\n================ 站立姿态搜索（零动作，看能活多少步）================")
        print(f"{'q1':>7} {'q2':>7} {'存活步数':>10} {'起始base_z':>12} {'起始pitch°':>12} {'base_z-轮心z':>14}")
        for cand in grid_candidates():
            res = measure(env, cand, args_cli.steps, args_cli.settle)
            results.append(res)
            print(
                f"{cand['L_joint1']:>7.2f} {cand['L_joint2']:>7.2f} {res['steps_survived']:>10d} "
                f"{res['start']['base_z']:>12.4f} {res['start']['base_pitch_deg']:>12.2f} "
                f"{res['start']['base_z_minus_wheel_z']:>14.4f}"
            )
        results.sort(key=lambda r: r["steps_survived"], reverse=True)
        print("\n---- 排名前 3 ----")
        for res in results[:3]:
            print(f"  {res['pose']} → 存活 {res['steps_survived']} 步")
            print(f"     final: {json.dumps(res['final'], ensure_ascii=False)}")
    else:
        pose = build_pose_overrides(args_cli.pose)
        print(f"\n================ 零动作 rollout（pose={pose or 'asset 默认位形'}）================")
        res = measure(env, pose, args_cli.steps, args_cli.settle)
        results.append(res)
        print(f"  存活步数 : {res['steps_survived']} / {args_cli.steps}  (survived={res['survived']})")
        print(f"  起始状态 : {json.dumps(res['start'], ensure_ascii=False)}")
        print(f"  结束状态 : {json.dumps(res['final'], ensure_ascii=False)}")
        print("\n  前 5 帧轨迹:")
        for row in res["trajectory_head"]:
            print(
                f"    base_z={row['base_z']:.4f} pitch={row['base_pitch_deg']:+.2f}° "
                f"base_z-轮心z={row['base_z_minus_wheel_z']:.4f} joint={row['joint_pos']}"
            )
        if not res["survived"]:
            print("\n  ⚠ 未活满全程 → 这个位形不是稳定站立位形；用 --grid 或 --pose 换一个再试。")

    out_path = os.path.join(_REPO_ROOT, args_cli.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"asset": asset, "results": results}, fh, ensure_ascii=False, indent=2)
    print(f"\n>>> JSON 已写入 {out_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
