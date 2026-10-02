# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DeformableSuspensionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 20000
    save_interval = 500
    experiment_name = "deformable_suspension_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=4.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DeformableTransformerPolicyCfg(RslRlPpoActorCriticCfg):
    """单帧结构化 token Transformer（1 全局 + 4 腿 token），见 agent_rl ActorCriticTransformer。"""

    class_name: str = "ActorCriticTransformer"
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 128
    head_hidden: int = 64
    actor_head: str = "per_leg"  # 每腿 token → 该腿动作；"global" = 全局 token → 4 维
    # 基类必填字段（transformer 不使用 hidden_dims/activation）
    actor_hidden_dims: list = []
    critic_hidden_dims: list = []
    activation: str = "elu"


@configclass
class DeformableSuspensionTransformerPPORunnerCfg(DeformableSuspensionPPORunnerCfg):
    experiment_name = "deformable_suspension_transformer"
    policy = DeformableTransformerPolicyCfg(init_noise_std=1.0)

    def __post_init__(self):
        # transformer 在 5e-5 下学得过慢；adaptive 调度会按 KL 自动回调
        self.algorithm.learning_rate = 3.0e-4


@configclass
class DeformableHistoryTransformerPolicyCfg(DeformableTransformerPolicyCfg):
    min_noise_std: float = 0.03
    history_length: int = 8
    actor_layout: dict = {
        "global": list(range(10)) + [30, 31],
        "legs": [[10 + i, 14 + i, 18 + i, 22 + i, 26 + i] for i in range(4)],
    }
    critic_layout: dict = {
        "global": list(range(10)) + [30, 31, 32, 33, 34, 35],
        "legs": [[10 + i, 14 + i, 18 + i, 22 + i, 26 + i, 36 + i] for i in range(4)],
    }


@configclass
class DeformableDynamicPPORunnerCfg(DeformableSuspensionTransformerPPORunnerCfg):
    experiment_name = "deformable_minangle_residual_v2"
    max_iterations = 10000
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.3, noise_std_type="log",
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        self.algorithm.class_name = "DiagnosticPPO"
        self.algorithm.separate_grad_clip = True
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.value_loss_coef = 1.0
        self.algorithm.entropy_coef = 0.005
        self.algorithm.num_learning_epochs = 3


@configclass
class DeformableLegacyPPORunnerCfg(DeformableSuspensionTransformerPPORunnerCfg):
    experiment_name = "deformable_dynamic_low_slip_history_v1"
    clip_actions = 1.0
    policy = DeformableHistoryTransformerPolicyCfg(init_noise_std=0.4, min_noise_std=0.0)
