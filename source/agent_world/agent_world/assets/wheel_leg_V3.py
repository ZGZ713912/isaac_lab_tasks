# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V3 (闭链轮腿) asset configuration —— wheelbipe 训练栈专用。
#
# 与 Wheel_leg_V2.py 使用同一台机器人 USD 与同一套关节（闭链 4 spherical +
# 2 prismatic 气簧在 USD 内），但执行器改为 wheelbipe 基座使用的 IdealPD：
#   legs_act : 4 个腿电机关节，位置伺服（kp/kd），沿用 wheelbipe legs_act 值
#   wheel    : 2 个轮关节，速度伺服（damping），沿用 wheelbipe wheel 值
#   legs_inact: 12 个被动闭链关节，effort=0，仅由闭合约束带动
#
# 说明：气簧是 excludeFromArticulation 的 loop prismatic，不能下发 effort，
# 由环境按 BKB 力曲线手动施加轴向力（见 wheel_leg_v3 环境）。
# =============================================================================

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg

from agent_world import AssetPath
from agent_world.assets.wheel_leg_V2 import (
    DEFAULT_SPAWN_HEIGHT,
    DM8009_ARMATURE,
    LEGS_ACT_JOINT_NAMES,
    PASSIVE_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
)

SOLVER_VELOCITY_LIMIT = 1.0e9


WheelLegV3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/Wheel_leg_V2/Wheel_leg_V2.usd",
        activate_contact_sensors=True,
        copy_from_source=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            # 闭链 spherical 约束需要更多求解迭代；对齐 validate 脚本（16/8）
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, DEFAULT_SPAWN_HEIGHT),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 驱动腿关节：位置 PD（wheelbipe legs_act 同值）
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=LEGS_ACT_JOINT_NAMES,
            stiffness=60.0,
            damping=2.0,
            effort_limit=40.0,
            velocity_limit=17.0,
            armature=DM8009_ARMATURE,
        ),
        # 驱动轮：速度伺服（wheelbipe wheel 同值）
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            stiffness=0.0,
            damping=0.2,
            effort_limit=5.0,
            velocity_limit=60.0,
            armature=0.0,
        ),
        # 被动闭链关节：不驱动（effort 0），仅随闭合约束运动
        "legs_inact": IdealPDActuatorCfg(
            joint_names_expr=PASSIVE_JOINT_NAMES,
            stiffness=0.0,
            damping=0.0,
            effort_limit=0.0,
            velocity_limit=SOLVER_VELOCITY_LIMIT,
            armature=0.0,
        ),
    },
)
