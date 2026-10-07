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
    use_leg_geometry_features: bool = False
    previous_action_pair_filter: bool = False
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
class DeformablePrecisionPPORunnerCfg(DeformableDynamicPPORunnerCfg):
    experiment_name = "deformable_foundation_precision_v2"
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.15, noise_std_type="log",
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.001


@configclass
class DeformableLegacyPPORunnerCfg(DeformableSuspensionTransformerPPORunnerCfg):
    experiment_name = "deformable_dynamic_low_slip_history_v1"
    clip_actions = 1.0
    policy = DeformableHistoryTransformerPolicyCfg(init_noise_std=0.4, min_noise_std=0.0)


@configclass
class DeformableReal2SimPPORunnerCfg(DeformableDynamicPPORunnerCfg):
    experiment_name = "deformable_real2sim_current_v1"


@configclass
class DeformableReal2SimPrecisionPPORunnerCfg(DeformablePrecisionPPORunnerCfg):
    experiment_name = "deformable_real2sim_precision_current_v1"


@configclass
class DeformableFittedPPORunnerCfg(DeformableDynamicPPORunnerCfg):
    experiment_name = "deformable_real2sim_fitted_v3"


@configclass
class DeformableFittedPrecisionPPORunnerCfg(DeformablePrecisionPPORunnerCfg):
    experiment_name = "deformable_real2sim_fitted_precision_v3"


@configclass
class DeformableFittedAdaptivePPORunnerCfg(DeformableFittedPrecisionPPORunnerCfg):
    experiment_name = "deformable_real2sim_adaptive_v3"
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.1, noise_std_type="log", use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.003


@configclass
class DeformableFittedBalancedPPORunnerCfg(DeformableFittedAdaptivePPORunnerCfg):
    experiment_name = "deformable_real2sim_balanced_v3"
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.06, noise_std_type="log", use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.001


@configclass
class DeformableFittedMobilityPPORunnerCfg(DeformableFittedBalancedPPORunnerCfg):
    experiment_name = "deformable_real2sim_mobility_v3"
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.05, noise_std_type="log", use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)


@configclass
class DeformableFittedSafeMobilityPPORunnerCfg(DeformableFittedMobilityPPORunnerCfg):
    experiment_name = "deformable_real2sim_safe_mobility_v3"
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.04, noise_std_type="log", use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)


@configclass
class DeformableFittedNativePrecisionPPORunnerCfg(DeformableFittedSafeMobilityPPORunnerCfg):
    experiment_name = "deformable_real2sim_native_precision_v3"
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.01, min_noise_std=0.005, noise_std_type="log",
        use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0002
        self.algorithm.schedule = "adaptive"


@configclass
class DeformableFittedSteepExposurePPORunnerCfg(DeformableFittedSafeMobilityPPORunnerCfg):
    experiment_name = "deformable_real2sim_steep_exposure_v3"
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.03, min_noise_std=0.01, noise_std_type="log",
        use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0005


@configclass
class DeformableFittedTractionReservePPORunnerCfg(DeformableFittedSteepExposurePPORunnerCfg):
    experiment_name = "deformable_real2sim_traction_reserve_v3"


@configclass
class DeformableFittedLowProfilePPORunnerCfg(DeformableFittedTractionReservePPORunnerCfg):
    experiment_name = "deformable_real2sim_low_profile_v3"
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.015, min_noise_std=0.005, noise_std_type="log",
        use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0003


@configclass
class DeformableFittedTerrainLowProfilePPORunnerCfg(DeformableFittedLowProfilePPORunnerCfg):
    experiment_name = "deformable_real2sim_terrain_low_profile_v3"


@configclass
class DeformableFittedAnchoredLowProfilePPORunnerCfg(DeformableFittedTerrainLowProfilePPORunnerCfg):
    experiment_name = "deformable_real2sim_anchored_low_profile_v3"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.steep_preservation_weight = 1.0
        self.algorithm.steep_reference_start_deg = 3.0
        self.algorithm.steep_reference_full_deg = 8.0
        self.algorithm.steep_reference_action_scale = 0.03


@configclass
class DeformableFittedNativeLowProfilePPORunnerCfg(DeformableFittedAnchoredLowProfilePPORunnerCfg):
    experiment_name = "deformable_real2sim_native_low_profile_v3"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.flat_posture_weight = 1.0
        self.algorithm.flat_posture_target_deg = 19.0
        self.algorithm.flat_posture_tilt_deg = 3.0
        self.algorithm.flat_posture_max_spread_deg = 0.0
        self.algorithm.reference_all_postures = False


@configclass
class DeformableFittedMixedCornerPPORunnerCfg(DeformableFittedNativeLowProfilePPORunnerCfg):
    experiment_name = "deformable_real2sim_mixed_corner_v3"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.flat_posture_anchor = "lowest"
        self.algorithm.steep_reference_start_deg = 8.0
        self.algorithm.steep_reference_full_deg = 12.0


@configclass
class DeformableFittedLevelingPPORunnerCfg(DeformableFittedMixedCornerPPORunnerCfg):
    experiment_name = "deformable_real2sim_leveling_v3"
    max_iterations = 101
    num_steps_per_env = 24
    save_interval = 50
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.08, min_noise_std=0.03, noise_std_type="log",
        use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        super().__post_init__()
        # Fresh PPO exploration; no frozen old-policy targets constrain leveling.
        self.algorithm.steep_preservation_weight = 0.0
        self.algorithm.reference_all_postures = False
        self.algorithm.flat_posture_weight = 0.0
        self.algorithm.entropy_coef = 0.001


@configclass
class DeformableFittedSupportLevelingPPORunnerCfg(DeformableFittedLevelingPPORunnerCfg):
    experiment_name = "deformable_real2sim_support_leveling_five_v3"
    max_iterations = 201
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.05, min_noise_std=0.02, noise_std_type="log",
        use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5.e-5


@configclass
class DeformableFittedSupportLevelingMotionPPORunnerCfg(DeformableFittedSupportLevelingPPORunnerCfg):
    experiment_name = "deformable_real2sim_support_leveling_motion_v3"
    max_iterations = 401


@configclass
class DeformableFittedSupportLevelingTenPPORunnerCfg(DeformableFittedSupportLevelingPPORunnerCfg):
    experiment_name = "deformable_real2sim_support_leveling_ten_v3"
    max_iterations = 401
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.015, min_noise_std=0.005, noise_std_type="log",
        use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0003


@configclass
class DeformableFittedSupportLevelingMixedPPORunnerCfg(DeformableFittedSupportLevelingTenPPORunnerCfg):
    experiment_name = "deformable_real2sim_support_leveling_mixed_v3"
    max_iterations = 601
    policy = DeformableHistoryTransformerPolicyCfg(
        init_noise_std=0.015, min_noise_std=0.005, noise_std_type="log",
        use_leg_geometry_features=True,
        actor_obs_normalization=False, critic_obs_normalization=False)
