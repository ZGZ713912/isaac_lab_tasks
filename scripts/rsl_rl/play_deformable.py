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
    X/Z  自旋 +/-(ωz, 增量)   Q    切换高/低车身基准角   L  全部归零

用法（仓库根目录，需 GUI）：
    python scripts/rsl_rl/play_deformable.py \
        --task=Robotics-Deformable-Suspension-Rough-Keyboard-Play-v0 \
        --checkpoint=logs/rsl_rl/deformable_suspension_direct/<ts>/model_XXXX.pt \
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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard play for deformable active suspension.")
parser.add_argument(
    "--task",
    type=str,
    default="Robotics-Deformable-Suspension-Rough-Keyboard-Play-v0",
    help="Task name (keyboard play variant).",
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXXX.pt")
parser.add_argument("--num_envs", type=int, default=1, help="Must be 1 for keyboard play.")
parser.add_argument("--vx_max", type=float, default=3.0, help="Max forward/backward speed (m/s).")
parser.add_argument("--vy_max", type=float, default=1.6, help="Max lateral speed (m/s).")
parser.add_argument("--wz_max", type=float, default=4.5, help="Max yaw rate (rad/s).")
parser.add_argument("--wz_step", type=float, default=0.6, help="Yaw-rate increment per Z/X press (rad/s).")
parser.add_argument("--no_vis", action="store_true", default=False, help="Disable play visualization (contact/level/HUD).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402

import agent_world  # noqa: F401,E402
import agent_tasks  # noqa: F401,E402
import cli_args as rsl_cli_args  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from keyboard_controller_deformable import DeformableKeyboard, DeformableKeyboardCfg  # noqa: E402
from deformable_play_vis import DeformablePlayVis  # noqa: E402
from agent_tasks.direct.deformable_suspension import cfg_utils as du  # noqa: E402

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


def main() -> None:
    _disable_viewport_wasd()
    # ---- env + agent config -------------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.play = True

    ns = argparse.Namespace(
        task=args_cli.task,
        device=args_cli.device,
        seed=None,
        run_name=None,
        logger=None,
        log_project_name=None,
        clip_actions=None,
        cmoe_router_temperature=None,
        moe_load_balancing_coef=None,
        cmoe_aux=None,
        experiment_name=None,
        resume=None,
        load_run=None,
        checkpoint=None,
    )
    agent_cfg = rsl_cli_args.parse_rsl_rl_cfg(args_cli.task, ns)
    agent_cfg.device = args_cli.device

    # ---- environment --------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    # env 创建后 viewport 已就绪，再确保一次 orbit（防 App 启动时未生效）
    _disable_viewport_wasd()

    keyboard = DeformableKeyboard(
        DeformableKeyboardCfg(
            vx_max=args_cli.vx_max,
            vy_max=args_cli.vy_max,
            wz_max=args_cli.wz_max,
            wz_step=args_cli.wz_step,
            q_choices=(du.Q_LOW,),
        )
    )
    keyboard.set_env(env)
    print(f"[INFO] {keyboard}")

    # ---- visualization (A 接触 + B 水平 + C HUD) ----
    play_vis = None
    if not args_cli.no_vis and not getattr(args_cli, "headless", False):
        play_vis = DeformablePlayVis(env)
        print("[INFO] 可视化已开启：红球=接触, 绿杆=世界竖直/红杆=车身z轴, HUD=姿态/高度/接触力")

    # ---- runner + checkpoint ------------------------------------------------
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- reset + loop -------------------------------------------------------
    obs, _ = env.reset()
    keyboard.reset()
    print("[INFO] 键盘 play 已启动：W/S 前后，A/D 横移，X/Z 自旋，Q 切换高低车身，L 归零")
    print("[INFO] 请确保 Isaac Sim 窗口有焦点才能接收键盘输入")
    print("[INFO] 相机自动跟随；WASD 已留给机器人（视口 orbit，不再 fly）")

    while simulation_app.is_running():
        keyboard.apply()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        if play_vis is not None:
            play_vis.update()
        if not getattr(args_cli, "headless", False):
            camera_follow(env)
        if bool(dones.any()):
            keyboard.reset()
            print("[INFO] 环境重置，键盘命令已归零")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
