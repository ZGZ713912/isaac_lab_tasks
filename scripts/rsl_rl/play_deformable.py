# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
# =============================================================================

"""Deformable 主动悬挂键盘 play：加载 checkpoint + GUI + 键盘遥控查看训练效果。

键位：
    W/S  前进/后退 (vx)      A/D  左移/右移 (vy)
    X/Z  自旋 +/-(ωz, 增量)   Q    查询/切换基准角   L  全部归零

用法（仓库根目录，需 GUI）：
    python scripts/rsl_rl/play_deformable.py \
        --task=Robotics-Deformable-Suspension-Rough-Keyboard-Play-History-Transformer-v2 \
        --checkpoint=logs/rsl_rl/deformable_foundation_precision_v2/<run>/model_999.pt \
        --grade-deg=20 --fixed_camera --debug_motion \
        --device=cuda:0
"""

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
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard play for deformable active suspension.")
parser.add_argument(
    "--task",
    type=str,
    default="Robotics-Deformable-Suspension-Rough-Keyboard-Play-History-Transformer-v2",
    help="Task name (keyboard play variant).",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXXX.pt")
parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 for keyboard play.")
parser.add_argument("--vx_max", type=float, default=3.0, help="Max forward/backward speed (m/s).")
parser.add_argument("--vy_max", type=float, default=1.6, help="Max lateral speed (m/s).")
parser.add_argument("--wz_max", type=float, default=6.283185307179586, help="Max yaw rate (rad/s); 2*pi = 60 rpm.")
parser.add_argument("--wz_step", type=float, default=0.6, help="Yaw-rate increment per Z/X press (rad/s).")
parser.add_argument("--no_vis", action="store_true", default=False, help="Disable play visualization (contact/level/HUD).")
parser.add_argument("--fixed_camera", action="store_true", help="Keep the camera stationary between resets to see world displacement.")
parser.add_argument("--debug_motion", action="store_true", help="Print commanded and measured motion every 0.5 simulation seconds.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many policy steps; 0 runs until closed.")
parser.add_argument(
    "--grade-deg",
    type=float,
    default=None,
    help="Override the periodic terrain with a constant 0..20 degree ramp for visual validation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.num_envs != 1 or args_cli.max_steps < 0:
    parser.error("keyboard play requires --num_envs=1 and nonnegative --max_steps")
if not os.path.isfile(args_cli.checkpoint):
    parser.error("checkpoint file does not exist")

sys.argv = [sys.argv[0]] + hydra_args
from play_lifecycle import PlayLifecycle

shutdown = PlayLifecycle()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
shutdown.bind_app(simulation_app)

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402

import agent_world  # noqa: F401,E402
import agent_tasks  # noqa: F401,E402
import agent_rl.rsl_rl.modules  # noqa: F401,E402  注册 ActorCriticTransformer 到 OnPolicyRunner
import agent_rl.rsl_rl.algorithms  # noqa: F401,E402  注册 DiagnosticPPO 到 OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from keyboard_controller_deformable import DeformableKeyboard, DeformableKeyboardCfg  # noqa: E402
from deformable_play_vis import DeformablePlayVis  # noqa: E402
from agent_tasks.direct.deformable_suspension import cfg_utils as du  # noqa: E402
from scripts.utils.deformable_checkpoint import validate_deformable_checkpoint  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _disable_viewport_wasd() -> None:
    """把视口相机从 fly 切到 orbit，避免 WASD 抢机器人遥控键。

    Kit 的 viewportMode 格式是 [viewport_id, "fly"|"orbit"]（见
    omni.kit.viewport.window tests）。纯字符串 set 不会真正切模式。
    """
    try:
        import carb

        settings = carb.settings.get_settings()
        viewport_id = None
        try:
            from omni.kit.viewport.utility import get_active_viewport

            vp = get_active_viewport()
            viewport_id = getattr(vp, "viewport_id", None) or getattr(vp, "id", None)
        except Exception:  # noqa: BLE001
            viewport_id = None

        if viewport_id is not None:
            settings.set(
                "/exts/omni.kit.manipulator.camera/viewportMode",
                [viewport_id, "orbit"],
            )
            print(f"[INFO] 视口相机已切到 orbit（viewport_id={viewport_id}，禁用 WASD fly）")
        else:
            # 兜底：部分版本接受全局字符串
            settings.set("/exts/omni.kit.manipulator.camera/viewportMode", "orbit")
            print("[INFO] 视口相机已设为 orbit（全局，未取到 viewport_id）")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 无法切换视口相机模式: {exc}")


def camera_follow(env) -> None:
    """相机跟随机器人（移植 play.py，减少手动转视角需求）。"""
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]
    camera_offset = torch.tensor([-3.0, 0.0, 0.5], dtype=torch.float32, device=env.device)
    camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
    ).squeeze(0)
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy()
    )


def _configure_grade(env_cfg, grade_deg: float | None) -> None:
    """Make a long, constant ramp before Isaac Lab builds the terrain."""
    if grade_deg is None:
        return
    if not math.isfinite(grade_deg) or not 0.0 <= grade_deg <= 20.0:
        raise ValueError("--grade-deg must be finite and in [0, 20]")
    generator = getattr(env_cfg.terrain, "terrain_generator", None)
    sub_terrains = getattr(generator, "sub_terrains", None) if generator is not None else None
    sub = sub_terrains.get("periodic_slope") if sub_terrains else None
    if sub is None:
        raise ValueError("--grade-deg requires a periodic_slope terrain task")
    # A 20 m segment keeps the vehicle on one uphill ramp instead of hiding the
    # behavior at a short crest/flat transition.
    sub.angle_range = (grade_deg, grade_deg)
    sub.angle_choices = None
    sub.segment_length = 20.0
    env_cfg.boundary_reset_enabled = False
    env_cfg.spawn_dir_jitter = False
    env_cfg.spawn_phase_stratify = False
    if grade_deg > 5.0:
        # A horizontal, four-contact reset may be impossible on a steep ramp.
        env_cfg.best_effort_leveling = True
        env_cfg.best_effort_tilt_weight = 6.0
        env_cfg.termination_roll_deg = 45.0
        env_cfg.termination_pitch_deg = 45.0


def _place_on_grade_ramp(env) -> None:
    """Put the single play environment well inside the uphill part of the ramp."""
    unwrapped = env.unwrapped
    if not getattr(unwrapped, "_periodic", False):
        return
    # The periodic profile starts at x + _profile_x_offset.  Phase 90 m with a
    # 20 m segment is 10 m into the 0..20 m uphill section.
    origins = unwrapped.scene.env_origins
    origins[:, 0] = 90.0 - float(unwrapped._profile_x_offset)
    origins[:, 1] = 0.0
    origins[:, 2] = 0.0


def _print_suspension_reference(env) -> None:
    """Print the configured q reference and its physical/height interpretation."""
    unwrapped = env.unwrapped
    q_choices = tuple(float(q) for q in getattr(unwrapped.cfg, "q_cmd_choices", ()))
    q_cmd = float(unwrapped.q_cmd[0].item()) if hasattr(unwrapped, "q_cmd") else None
    if not q_choices and q_cmd is None:
        return
    q_ref = q_cmd if q_cmd is not None else q_choices[0]
    zero = float(getattr(unwrapped.cfg,"leg_physical_angle_zero",du.PHYSICAL_MAX_ANGLE))
    q_limit = float(unwrapped.cfg.leg_target_upper_limit)
    print(
        "[INFO] 悬挂基准: "
        f"q_cmd={q_ref:.4f} rad, physical={math.degrees(zero-q_ref):.2f} deg, "
        f"base_h={float(du.q_to_base_height(q_ref)):.4f} m; "
        f"q_target_max={q_limit:.4f} rad, base_h={float(du.q_to_base_height(q_limit)):.4f} m",
        flush=True,
    )
    print(
        "[INFO] q 越大车体越低；q_cmd 是零动作基准，q_target_max 是目标范围上限，"
        "不是同一个量。",
        flush=True,
    )


def main() -> None:
    _disable_viewport_wasd()
    # ---- env + agent config -------------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.play = True
    _configure_grade(env_cfg, args_cli.grade_deg)
    resume_path = os.path.abspath(args_cli.checkpoint)
    validate_deformable_checkpoint(env_cfg, resume_path)
    # Match the initial visible asset to the configured low-body reference.
    # Reset still solves per-corner contact on terrain before the first action.
    q_ref = float(env_cfg.q_cmd_choices[0])
    env_cfg.robot_cfg.init_state.joint_pos = {
        **{name: q_ref for name in du.ORDERED_LEG_JOINT_NAMES + du.ORDERED_WS_JOINT_NAMES},
        **{name: -q_ref for name in du.ORDERED_UPPER_LEG_JOINT_NAMES},
        **{name: 0.0 for name in du.ORDERED_WHEEL_JOINT_NAMES},
    }
    initial_pos = env_cfg.robot_cfg.init_state.pos
    env_cfg.robot_cfg.init_state.pos = (
        initial_pos[0], initial_pos[1],
        float(du.q_to_base_height(q_ref)) + max(0.0, env_cfg.reset_height_buffer),
    )

    from scripts.utils.deformable_checkpoint import load_deformable_agent_config
    agent_cfg = load_deformable_agent_config(resume_path, args_cli.device)
    if hasattr(env_cfg, "policy_history_length"):
        env_cfg.policy_history_length = agent_cfg["policy"].get("history_length", 1)
        env_cfg.observation_space = 32 * env_cfg.policy_history_length

    # ---- environment --------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    shutdown.bind_env(env)
    if args_cli.grade_deg is not None:
        _place_on_grade_ramp(env)
        print(f"[INFO] 固定坡度 play: {args_cli.grade_deg:.1f}°，已放置在 20 m 上坡段内部", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
    obs, _ = env.reset()
    _print_suspension_reference(env)
    q = env.unwrapped.robot.data.joint_pos[0, env.unwrapped._legs_idx]
    print(f"[INFO] 策略运行前初始腿角 q={q.tolist()}", flush=True)
    # env 创建后 viewport 已就绪，再确保一次 orbit（防 App 启动时未生效）
    _disable_viewport_wasd()

    keyboard = None if args_cli.headless else DeformableKeyboard(
        DeformableKeyboardCfg(
            vx_max=args_cli.vx_max,
            vy_max=args_cli.vy_max,
            wz_max=args_cli.wz_max,
            wz_step=args_cli.wz_step,
            q_choices=tuple(env.unwrapped.cfg.q_cmd_choices),
        )
    )
    if keyboard is not None:
        keyboard.set_env(env)
        keyboard.reset()
        print(f"[INFO] {keyboard}")

    # ---- visualization (A 接触 + B 水平 + C HUD) ----
    play_vis = None
    if not args_cli.no_vis and not getattr(args_cli, "headless", False):
        play_vis = DeformablePlayVis(env)
        print("[INFO] 可视化已开启：红球=接触, 绿杆=世界竖直/红杆=车身z轴, HUD=姿态/高度/接触力")

    # ---- runner + checkpoint ------------------------------------------------
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=args_cli.device)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- reset + loop -------------------------------------------------------
    camera_follow.smooth_camera_positions = []
    if not getattr(args_cli, "headless", False):
        camera_follow(env)
    print("[INFO] play 已启动：W/S 前后，A/D 横移，X/Z 自旋，Q 查询/切换基准，L 归零")
    print("[INFO] 请确保 Isaac Sim 窗口有焦点才能接收键盘输入")
    print(f"[INFO] 相机模式：{'固定（可观察世界位移）' if args_cli.fixed_camera else '自动跟随（机器人会留在画面中央）'}")
    motion_tick = 0
    motion_interval = max(1, round(0.5 / env.unwrapped.step_dt))
    motion_origin = env.unwrapped.robot.data.root_pos_w[0].clone()

    while simulation_app.is_running() and not shutdown.requested:
        if keyboard is not None:
            keyboard.apply()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        if play_vis is not None:
            play_vis.update()
        if not getattr(args_cli, "headless", False) and not args_cli.fixed_camera:
            camera_follow(env)
        motion_tick += 1
        if args_cli.debug_motion and motion_tick % motion_interval == 0:
            u = env.unwrapped
            data = u.robot.data
            cmd = u.cmd_buf[0].tolist()
            vel = data.root_link_lin_vel_b[0, :2].tolist()
            delta = (data.root_pos_w[0, :2] - motion_origin[:2]).tolist()
            q = data.joint_pos[0, u._legs_idx]
            target = u.leg_target[0]
            physical_zero = float(getattr(u.cfg, "leg_physical_angle_zero", du.PHYSICAL_MAX_ANGLE))
            physical_angles = torch.rad2deg(physical_zero - q).tolist()
            gravity = data.projected_gravity_b[0]
            tilt = math.degrees(math.atan2(float(gravity[:2].norm()), -float(gravity[2])))
            loads = u.wheel_normal_forces[0]
            contacts = int((loads > u.cfg.wheel_contact_force_threshold).sum())
            print(
                f"[motion] cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                f"vel_b=({vel[0]:+.3f},{vel[1]:+.3f}) "
                f"wz={data.root_ang_vel_b[0, 2].item():+.3f} "
                f"delta_xy_w=({delta[0]:+.3f},{delta[1]:+.3f}) m "
                f"q=[{q.min().item():.3f},{q.max().item():.3f}] "
                f"target=[{target.min().item():.3f},{target.max().item():.3f}] "
                f"physical_deg=[{','.join(f'{angle:.1f}' for angle in physical_angles)}] "
                f"body_top={u.body_top_height[0].item() * 1000:.1f}mm "
                f"q_cmd={u.q_cmd[0].item():.3f} tilt={tilt:.2f}deg "
                f"contact={contacts}/4 min_load={loads.min().item():.1f}N "
                f"slip={u._last_wheel_slip[0].abs().mean().item():.3f}m/s",
                flush=True,
            )
        if bool(dones.any()):
            if keyboard is not None:
                keyboard.reset()
            motion_origin = env.unwrapped.robot.data.root_pos_w[0].clone()
            camera_follow.smooth_camera_positions = []
            if not getattr(args_cli, "headless", False):
                camera_follow(env)
            print("[INFO] 环境重置，键盘命令已归零")
        if args_cli.max_steps and motion_tick >= args_cli.max_steps:
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(">>> interrupted by user, closing", flush=True)
    finally:
        shutdown.close()
