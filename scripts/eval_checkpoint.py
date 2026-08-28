# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# =============================================================================

"""无头定量评估脚本：运行训练好的 checkpoint，输出量化指标，不需要 GUI/渲染器。

用法（仓库根目录）：
  python scripts/eval_checkpoint.py \
      --task=Robotics-Wheelbipe-V14-Flat-Play-v0 \
      --checkpoint=./logs/rsl_rl/wheelbipe_v14_2_flat_direct/<时间戳>/model_1999.pt \
      --num_envs=16 --episodes=200 --device=cuda:0 --headless

输出：终端汇总表 + CSV（默认 logs/debug/eval_<时间戳>.csv）。
指标：平均 episode 长度（步/秒）、存活率（time_out 占比）、平均 reward、
      速度跟踪误差（vx/vy/omega 的 |cmd-actual| 均值）、平均车体高度。
"""

"""Launch Isaac Sim Simulator first."""

import os
import sys

# ensure the repository root is importable regardless of how the script is invoked
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# scripts/rsl_rl/ holds cli_args.py (shared with train.py / play.py)
_RSL_RL_SCRIPTS = os.path.join(_REPO_ROOT, "scripts", "rsl_rl")
if _RSL_RL_SCRIPTS not in sys.path:
    sys.path.insert(0, _RSL_RL_SCRIPTS)

import argparse
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Headless evaluation of a trained checkpoint.")
parser.add_argument("--task", type=str, default="Robotics-Wheelbipe-V14-Flat-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXXX.pt")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=200, help="Total episodes to evaluate")
parser.add_argument("--max_steps", type=int, default=0, help="Cap steps per episode (0 = env max length)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--csv", type=str, default=None, help="CSV output path (default: logs/debug/eval_<ts>.csv)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# always headless (evaluation needs no renderer)
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import csv
import statistics

import gymnasium as gym
import torch

import agent_world  # noqa: F401
import agent_tasks  # noqa: F401
import cli_args as rsl_cli_args
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    # ---- configuration -----------------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.play = True
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    ns = argparse.Namespace(
        task=args_cli.task,
        device=args_cli.device,
        seed=args_cli.seed,
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

    # ---- environment + runner ---------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {os.path.abspath(args_cli.checkpoint)}")
    runner.load(os.path.abspath(args_cli.checkpoint), load_optimizer=False, map_location=env.device)
    policy = runner.get_inference_policy(device=env.device)

    unwrapped = env.unwrapped
    max_ep_len = int(env.max_episode_length)
    step_cap = args_cli.max_steps if args_cli.max_steps > 0 else max_ep_len

    # ---- evaluation loop ---------------------------------------------------
    episodes_done = 0
    ep_lengths = []          # steps
    ep_rewards = []          # total reward per episode
    terminations = {"terminate": 0, "time_out": 0, "max_steps": 0}
    vel_err = {"vx": [], "vy": [], "omega_z": []}
    heights = []
    total_steps = 0

    # per-env running accumulators
    num_envs = env.num_envs
    cur_reward = torch.zeros(num_envs, device=env.device)
    cur_len = torch.zeros(num_envs, dtype=torch.long, device=env.device)

    obs, extras = env.reset()
    print(f"[INFO] Evaluating up to {args_cli.episodes} episodes (max {step_cap} steps each, "
          f"{num_envs} envs) ...")

    hard_stop = False
    with torch.inference_mode():
        while episodes_done < args_cli.episodes and not hard_stop:
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)

            rewards = rewards.to(env.device)
            dones = dones.to(env.device)

            cur_reward += rewards
            cur_len += 1
            total_steps += num_envs

            # velocity tracking error (only while commanded)
            try:
                cmd = unwrapped.command.clone()
                lin_vel = unwrapped.robot.data.root_lin_vel_b
                ang_vel_z = unwrapped.robot.data.root_ang_vel_b[:, 2]
                vel_err["vx"].extend((cmd[:, 0] - lin_vel[:, 0]).abs().tolist())
                vel_err["vy"].extend((cmd[:, 1] - lin_vel[:, 1]).abs().tolist())
                vel_err["omega_z"].extend((cmd[:, 2] - ang_vel_z).abs().tolist())
                heights.extend(unwrapped.robot.data.root_pos_w[:, 2].tolist())
            except Exception:
                pass

            # collect finished episodes
            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            for i in done_ids.tolist():
                ep_lengths.append(int(cur_len[i].item()))
                ep_rewards.append(float(cur_reward[i].item()))
                if ep_lengths[-1] >= step_cap:
                    terminations["max_steps"] += 1
                elif bool(extras.get("time_outs", torch.zeros(1))[i].item()):
                    terminations["time_out"] += 1
                else:
                    terminations["terminate"] += 1
                cur_len[i] = 0
                cur_reward[i] = 0.0
                episodes_done += 1

            if total_steps > args_cli.episodes * (step_cap + 10) * 2:
                print("[WARN] Safety cap reached, stopping.")
                hard_stop = True

    # ---- summary -----------------------------------------------------------
    n = len(ep_lengths)
    print("\n" + "=" * 60)
    print(f"  Evaluation: {n} episodes  ({num_envs} parallel envs, device={args_cli.device})")
    print("=" * 60)
    if n == 0:
        print("  No episodes completed.")
        return
    ep_sec = [l * unwrapped.step_dt for l in ep_lengths]
    surv = terminations["time_out"] + terminations["max_steps"]
    print(f"  Episode length : mean={statistics.mean(ep_lengths):.1f} steps "
          f"({statistics.mean(ep_sec):.2f} s), median={statistics.median(ep_lengths):.0f}, "
          f"max={max(ep_lengths)} / {max_ep_len}")
    print(f"  Survival rate  : {surv}/{n} = {100.0 * surv / n:.1f}% "
          f"(time_out={terminations['time_out']}, max_steps={terminations['max_steps']}, "
          f"fell={terminations['terminate']})")
    print(f"  Mean episode reward: {statistics.mean(ep_rewards):.2f}")
    if vel_err["vx"]:
        print(f"  Vel tracking mean |err|: vx={statistics.mean(vel_err['vx']):.3f} m/s, "
              f"vy={statistics.mean(vel_err['vy']):.3f} m/s, "
              f"omega_z={statistics.mean(vel_err['omega_z']):.3f} rad/s")
    if heights:
        print(f"  Mean base height: {statistics.mean(heights):.3f} m (min={min(heights):.3f}, max={max(heights):.3f})")

    # ---- CSV ---------------------------------------------------------------
    csv_path = args_cli.csv or os.path.join(
        "logs", "debug", f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "length_steps", "length_s", "reward", "outcome"])
        for i in range(n):
            outcome = "fell" if ep_lengths[i] < step_cap else "timeout"
            w.writerow([i + 1, ep_lengths[i], round(ep_sec[i], 3), round(ep_rewards[i], 3), outcome])
    print(f"\n  Per-episode detail saved to: {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
