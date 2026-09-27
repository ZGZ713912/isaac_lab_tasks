# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V2 专用 play：加载 checkpoint，按固定/采样指令跑策略。
#
# 现有 scripts/rsl_rl/play.py 深度依赖 wheel_leg_v1/deformable 专有属性
# （command_generator / height_cmd / spring_force / terrain ...），不适用于本
# DirectRLEnv；因此单独提供本入口（对标 V40 仓 scripts/play_v40.py）。
#
# 用法（仓库根目录；看画面去掉 --headless，用 ./run_gui.sh 包裹避免 Arch 上 GUI 崩溃）：
#   python scripts/rsl_rl/play_wheel_leg_v2.py \
#       --task=Robotics-Wheel-Leg-V2-Stand-Play-v0 \
#       --checkpoint=logs/rsl_rl/wheel_leg_v2_flat_direct/<ts>/model_XXXX.pt \
#       --device=cuda:0 --headless
#   # 指定指令（vx m/s, wz rad/s, height m）：
#   python scripts/rsl_rl/play_wheel_leg_v2.py --task=Robotics-Wheel-Leg-V2-Flat-Play-v0 \
#       --checkpoint=... --command 1.0 0.5 0.24
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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wheel_leg_V2 policy play.")
parser.add_argument("--task", type=str, default="Robotics-Wheel-Leg-V2-Stand-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXXX.pt")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--command", type=float, nargs=3, default=None, metavar=("VX", "WZ", "HEIGHT"),
                    help="Fixed (vx m/s, wz rad/s, height m); omit to use contract sampling.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.command is not None:
        env_cfg.evaluation_command = tuple(float(x) for x in args_cli.command)

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

    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        if bool(dones.any()):
            print("[INFO] env reset")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
