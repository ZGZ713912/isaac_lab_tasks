# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Deformable 阶段 4 评估：把训练好的策略与“零动作（冻结在 Q_LOW）”基线对比，
# 输出车身倾角、四轮接触率、底盘余量、存活等指标。
#
# 用法：
#   python scripts/tools/deformable_eval_tilt.py \
#       --task=Robotics-Deformable-Suspension-Rough-v0 \
#       --checkpoint=logs/rsl_rl/deformable_suspension_direct/<ts>/model_4999.pt \
#       --num_envs=256 --steps=600 --device=cuda:0
# =============================================================================
"""Phase-4 policy-vs-baseline tilt evaluation for the deformable suspension task."""

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

parser = argparse.ArgumentParser(description="Deformable tilt eval: policy vs zero-action baseline.")
parser.add_argument("--task", type=str, default="Robotics-Deformable-Suspension-Rough-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=600, help="Rollout steps after warmup.")
parser.add_argument("--warmup", type=int, default=200)
parser.add_argument("--seed", type=int, default=1234)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse as _argparse  # noqa: E402
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import agent_world  # noqa: F401,E402
import agent_tasks  # noqa: F401,E402
import cli_args as rsl_cli_args  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True


def _make_env(device: str):
    cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    cfg.play = True
    cfg.events = None  # 评估关域随机化，保证可比
    cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=cfg)
    return env


def _rollout(env, policy, steps: int, warmup: int) -> dict:
    unwrapped = env.unwrapped
    obs, _ = env.reset()
    acc = {"tilt": [], "pitch": [], "roll": [], "contact": [], "clear": [], "base": []}
    with torch.no_grad():
        for i in range(warmup + steps):
            if policy is None:
                actions = torch.zeros(env.num_envs, unwrapped.cfg.action_space, device=env.device)
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if i < warmup:
                continue
            pgb = unwrapped.robot.data.projected_gravity_b
            acc["pitch"].append(pgb[:, 0].abs().mean().item())
            acc["roll"].append(pgb[:, 1].abs().mean().item())
            tilt = torch.sqrt(torch.clamp(pgb[:, 0] ** 2 + pgb[:, 1] ** 2, min=0.0))
            acc["tilt"].append(tilt.mean().item())
            acc["contact"].append(
                (unwrapped.wheel_contact_forces > 1.0).float().mean().item()
            )
            acc["clear"].append(unwrapped.chassis_clearance.mean().item())
            acc["base"].append(unwrapped._base_contact.float().mean().item())
    return {k: sum(v) / max(len(v), 1) for k, v in acc.items()}


def main() -> None:
    env = _make_env(args_cli.device)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    ns = _argparse.Namespace(
        task=args_cli.task, device=args_cli.device, seed=args_cli.seed, run_name=None,
        logger=None, log_project_name=None, clip_actions=None, cmoe_router_temperature=None,
        moe_load_balancing_coef=None, cmoe_aux=None, experiment_name=None, resume=None,
        load_run=None, checkpoint=None,
    )
    agent_cfg = rsl_cli_args.parse_rsl_rl_cfg(args_cli.task, ns)
    agent_cfg.device = args_cli.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint), load_optimizer=False, map_location=env.device)
    policy = runner.get_inference_policy(device=env.device)

    def deg(x):
        return math.degrees(math.asin(min(1.0, max(-1.0, x))))

    print("\n================ Phase-4 tilt eval ================")
    print(f"task={args_cli.task}  ckpt={os.path.basename(args_cli.checkpoint)}  envs={args_cli.num_envs}  steps={args_cli.steps}")
    pol = _rollout(env, policy, args_cli.steps, args_cli.warmup)
    base = _rollout(env, None, args_cli.steps, args_cli.warmup)

    hdr = f"{'metric':<28}{'policy':>12}{'zero-action':>14}{'delta':>12}"
    print(hdr)
    print("-" * len(hdr))
    for key, label in (
        ("tilt", "body tilt (deg)"),
        ("pitch", "|pitch| (deg)"),
        ("roll", "|roll| (deg)"),
        ("contact", "4-wheel contact frac"),
        ("clear", "chassis clearance (m)"),
        ("base", "base contact frac"),
    ):
        p, b = pol[key], base[key]
        if key in ("tilt", "pitch", "roll"):
            p, b = deg(p), deg(b)
        print(f"{label:<28}{p:>12.4f}{b:>14.4f}{p-b:>12.4f}")
    print("===================================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
