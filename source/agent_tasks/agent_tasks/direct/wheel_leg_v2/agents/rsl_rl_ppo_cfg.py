# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V2 PPO 配置；移植自 V40 训练仓
# wheeled-biped-rl-train/src/wheeled_tasks/agents/v40_ppo_cfg.py。
#
# 关键点：
#   - num_steps_per_env=48：policy 100Hz 下 0.48s，与 V40/华南虎快照时钟对齐；
#   - 动作由控制器裁剪（clip_actions=None），PPO 似然用原始采样；
#   - 观测不做经验归一化，网络 [256,128,64] ELU。
# =============================================================================

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class WheelLegV2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    num_steps_per_env = 48
    max_iterations = 20000
    save_interval = 100
    experiment_name = "wheel_leg_v2_flat_direct"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.2,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        entropy_coef=0.005,
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
