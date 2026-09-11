# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V1 (轮腿 V1) asset configuration —— 结构镜像 wheelbipe_V14_2.py，
# 但去云台(gimbal)/弹簧(spring)/被动腿(legs_inact)（本机无这些部件）。
#
# USD：assets/usd_files/Wheel_leg_V1/Wheel_leg_V1.usd
#   （由 urdf_V4.0.urdf 经 Isaac Lab UrdfConverter 转换，joint drive: force / target none）
# 6 DOF（URDF 顺序）：L_joint1/L_joint2/L_joint3, R_joint1/R_joint2/R_joint3 —— 全 continuous。
# 角色映射（与 wheelbipe 同构，动作 6 维）：
#   legs_act = L/R_joint1 + L/R_joint2   （腿关节，位置 PD，kp/kd/effort 暂用 wheelbipe legs_act 值）
#   wheel    = L/R_joint3                （末端轮，速度伺服，同 wheelbipe wheel 组）
# 无被动关节（legs_inact 为空）—— 相应事件表/索引在 env 侧已适配。
#
# 名义站姿：URDF 零位（q=0，SolidWorks 装配位形）；spawn z 按几何粗估，视检后调整。
# =============================================================================

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg

from agent_world import AssetPath

DM8009_ARMATURE = 1.95e-04 * 9.0 * 9.0  # 与 wheelbipe legs_act 一致


WheelLegV1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/Wheel_leg_V1/Wheel_leg_V1.usd",
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
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=6,
        ),
    ),
    # 零位关节角 = SolidWorks 装配位形（视为站立位形）；spawn z 高于轮触地高度，
    # 让车体轻微下落（数值待 view_robot 视检校准）。
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 腿关节：位置 PD（kp=60/kd=2/effort 40/vel 17 —— wheelbipe legs_act 同值）
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=[
                "L_joint1",
                "L_joint2",
                "R_joint1",
                "R_joint2",
            ],
            stiffness=60.0,
            damping=2.0,
            effort_limit=40.0,
            velocity_limit=17.0,
            armature=DM8009_ARMATURE,
        ),
        # 轮：速度伺服（damping 0.2/effort 5/vel 60 —— wheelbipe wheel 同值）
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=["L_joint3", "R_joint3"],
            stiffness=0.0,
            damping=0.2,
            effort_limit=5.0,
            velocity_limit=60.0,
            armature=0.0,
        ),
    },
)
