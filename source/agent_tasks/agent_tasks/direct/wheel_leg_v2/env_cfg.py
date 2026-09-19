# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V2 平地 DirectRLEnv 配置。
#
# 架构移植自 V40 训练仓 wheeled-biped-rl-train/src/wheeled_tasks/direct/v40_serial/env_cfg.py：
# 环境的具体维度/时钟/资产在 WheelLegV2Env.__init__ 里按合同（contract）填充。
# 合同动作和关节状态使用真实电机树关节：髋 L/R、膝 LL/RR、轮 L/R；
# 闭链输出由 USD 中的 spherical 约束传递，不再把主链膝关节当作电机输入。
# =============================================================================

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .contract import DEFAULT_CONTRACT, ROUND2_CONTRACT


@configclass
class WheelLegV2EnvCfg(DirectRLEnvCfg):
    """Wheel_leg_V2 平地任务的基类；子类只改 stage。"""

    decimation = 2
    episode_length_s = 20.0
    is_finite_horizon = False
    # [L_joint1, LL_joint1, L_joint3, R_joint1, RR_joint1, R_joint3]
    action_space = 6
    # DirectRLEnv 2.3 会把 observation_space 包成 "policy"，state_space 包成 "critic"。
    observation_space = 125
    state_space = 29
    sim = SimulationCfg(dt=0.005, render_interval=2)
    # 实测（/tmp 探测脚本）：
    #   replicate_physics=True + clone_in_fabric=False → 闭链 loop joints 被正确复制，
    #     多环境闭合误差 <0.001mm；这是可扩到 4096 env 的关键。
    #   clone_in_fabric=True → 接触传感器初始化失败（Failed to initialize contact
    #     reporter），必须保持 False。
    scene = InteractiveSceneCfg(
        num_envs=256, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False,
    )

    # 以下字段在 DirectRLEnv 构造场景之前由 __init__ 按合同填充
    robot_cfg: ArticulationCfg | None = None
    contact_sensor_cfg: ContactSensorCfg | None = None
    contract_path: str | None = str(DEFAULT_CONTRACT)
    stage: str = "stand"

    # 闭链误差看门狗（米）：任一闭合点两杆世界点距离超过该值即安全 reset；0 关闭。
    closure_error_tolerance: float = 0.005

    # 评估/演示：固定 (vx, wz, height) 覆盖采样命令；None = 正常训练采样。
    evaluation_command: tuple[float, float, float] | None = None


@configclass
class WheelLegV2StandEnvCfg(WheelLegV2EnvCfg):
    stage = "stand"


@configclass
class WheelLegV2HeightEnvCfg(WheelLegV2EnvCfg):
    stage = "height"


@configclass
class WheelLegV2FlatEnvCfg(WheelLegV2EnvCfg):
    stage = "locomotion"


@configclass
class WheelLegV2FlatRound2EnvCfg(WheelLegV2FlatEnvCfg):
    """V40 完成 profile：观测噪声 + 持续倾倒终止 + reset 根速度随机。"""
    contract_path = str(ROUND2_CONTRACT)


@configclass
class WheelLegV2FlatRound2PlayEnvCfg(WheelLegV2FlatRound2EnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False)


@configclass
class WheelLegV2StandPlayEnvCfg(WheelLegV2StandEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False)


@configclass
class WheelLegV2FlatPlayEnvCfg(WheelLegV2FlatEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False)
