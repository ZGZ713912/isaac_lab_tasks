# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Deformable suspension chassis (omni-b) asset configuration.
# USD 由 scripts/tools/convert_deformable_urdf.sh 从 deformable_infantry URDF 转换生成
# （12 DOF：joint_leg_1..4 + joint_wheel_set_1..4 + joint_wheel_1..4，
#   leg 与 wheel_set 平行四边形耦合，单电机驱动）。
#
# 所有执行器 stiffness/damping = 0（effort 模式）：
# 腿的 PD（kp=200/kd=4）与平四耦合弹簧在 env 内手工计算后 set_joint_effort_target。
# =============================================================================

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from agent_world import AssetPath

"""Configuration for the deformable suspension chassis (deformable_infantry)."""

DeformableSuspensionCFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/deformable_suspension/deformable_suspension.usd",
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
    # 名义站姿：q=0（四腿同角 0，车体原点离地 0.132 m，轮半径 0.077 m）
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.137),
        joint_pos={
            "joint_leg_.*": 0.0,
            "joint_wheel_set_.*": 0.0,
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
