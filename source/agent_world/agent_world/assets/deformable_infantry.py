# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Deformable (变形底盘) chassis asset configuration.
# USD：assets/usd_files/deformable_infantry/deformable_infantry.usd
#   （由 deformable_infantry/urdf/deformable_infantry.urdf 经 Isaac Lab
#     UrdfConverter 转换，joint drive: force / target none / convex_hull）
# 12 DOF：joint_leg_1..4 + joint_wheel_set_1..4 + joint_wheel_1..4，
# leg 与 wheel_set 平行四边形耦合（单电机驱动，耦合在 env 内以虚拟弹簧模拟）。
#
# 所有执行器 stiffness/damping = 0（effort 模式）：
# 腿的 PD（kp=200/kd=4）与平四耦合弹簧在 env 内手工计算后 set_joint_effort_target。
# =============================================================================

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from agent_world import AssetPath

"""Configuration for the deformable chassis (deformable_infantry)."""

DeformableInfantryCFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/deformable_infantry/deformable_infantry.usd",
        activate_contact_sensors=True,
        copy_from_source=True,  # Required for proper articulation loading in Isaac Lab 2.3.0
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
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    # 名义站姿：q = 1.3439（77°，max 位，腿与水平夹角 77°，车体最高）——
    # 角度约定已与部署统一：q 数值 = 最粗杆与水平面的夹角（5°~77°，量角器实测）。
    # spawn pos.z = 0.2：比名义站姿接触高度（~0.137）略高，让车体从高处轻微下落，
    # 避免初始过低导致轮子/腿部穿透地面而卡死（env 每次 reset 用 init_root_height=0.18）。
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            "joint_leg_.*": 1.3439,
            "joint_wheel_set_.*": 1.3439,
            "joint_wheel_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["joint_leg_.*"],
            stiffness=0.0,
            damping=0.0,
        ),
        "wheel_set": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_set_.*"],
            stiffness=0.0,
            damping=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
