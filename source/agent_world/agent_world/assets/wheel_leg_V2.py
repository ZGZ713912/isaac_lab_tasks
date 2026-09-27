# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V2 (闭链轮腿) asset configuration.
#
# USD：assets/usd_files/Wheel_leg_V2/Wheel_leg_V2.usd
#   （由 scripts/tools/build_wheel_leg_v2_closed_usd.py 直接 authored，
#    参考 闭链参考/urdf/tools/build_chassis_closedchain.py 的 reference-style：
#      树关节 = UsdPhysics.RevoluteJoint（axis 局部 Z）
#      四杆闭合 = UsdPhysics.SphericalJoint + excludeFromArticulation
#      气弹簧   = UsdPhysics.PrismaticJoint
#    闭链误差由 scripts/tools/validate_wheel_leg_v2_closed.py 校验。）
#
# 关节：18 个树 revolute + 4 个 spherical 闭合 + 2 个 prismatic 气弹簧。
#   - 驱动关节（真实电机所在的树关节）：
#       髋 L_joint1/R_joint1，膝 LL_joint1/RR_joint1，轮 L_joint3/R_joint3
#   - 被动关节：主链膝 L_joint2/R_jonit2、其余四杆关节和气弹簧支链关节。
#     不设驱动，仅由闭合约束（spherical + prismatic）跟随。
#
# 执行器：effort 模式（stiffness/damping=0），由 env 手工计算 PD/轮曲线力矩后
# set_joint_effort_target；轮用转速查 torque-speed envelope（同 V40/m3508）。
# 被动关节 effort 上限 0，保持自由。
#
# 名义位形：URDF 零位 q=0 = SolidWorks 装配位形（闭链在 q=0 自洽）。
# =============================================================================

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from agent_world import AssetPath

# 与 wheelbipe / wheel_leg_V1 同款达妙 DM8009 折算到关节的等效转子惯量
DM8009_ARMATURE = 1.95e-04 * 9.0 * 9.0

# 合同定义的 4 个腿部电机关节；动作交错顺序由 contract.py 统一定义。
LEGS_ACT_JOINT_NAMES = ["L_joint1", "LL_joint1", "R_joint1", "RR_joint1"]
WHEEL_JOINT_NAMES = ["L_joint3", "R_joint3"]
# 必须显式列出被动关节，避免 LL_joint1/RR_joint1 被通配符重复匹配。
PASSIVE_JOINT_NAMES = [
    "L_joint2", "R_jonit2",
    "LL_joint2", "LL_joint3", "LL_link4",
    "RR_joint2", "RR_joint3", "RR_joint4",
    "LLL_joint2", "LLL_jointt1", "RRR_joint2", "RRR_joint1",
]

LEG_EFFORT_LIMIT = 40.0      # N·m（DM-J8009 研究先验，非实测额定）
WHEEL_EFFORT_LIMIT = 3.837686567164179  # N·m（M3508 11:1 曲线 + effort 上限，同 V40）
SOLVER_VELOCITY_LIMIT = 1.0e9  # 关闭 URDF/UUSD 占位速度上限；真实由轮曲线/腿限幅约束

# 重置时的默认根高（米）。env 每次 reset 会用合同 nominal_base_height 覆盖；
# 这里仅作为 ArticulationCfg 的初始 spawn。接入气弹簧（BKB0.45-063-172, 365N 预载）后
# 实测零动作静平衡高 ≈0.231 m，合同取 0.23。
DEFAULT_SPAWN_HEIGHT = 0.23


WheelLegV2_CFG = ArticulationCfg(
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
        # 驱动腿关节：effort 模式，PD 在 env 内算
        "legs_act": ImplicitActuatorCfg(
            joint_names_expr=LEGS_ACT_JOINT_NAMES,
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=LEG_EFFORT_LIMIT,
            velocity_limit_sim=SOLVER_VELOCITY_LIMIT,
            armature=DM8009_ARMATURE,
            friction=0.0,
        ),
        # 驱动轮：effort 模式，轮速误差经 kd + torque-speed 曲线求力矩
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=WHEEL_EFFORT_LIMIT,
            velocity_limit_sim=SOLVER_VELOCITY_LIMIT,
            armature=0.0,
            friction=0.0,
        ),
        # 被动闭链关节：不驱动（effort 0），仅随闭合约束运动
        "passive_leg": ImplicitActuatorCfg(
            joint_names_expr=PASSIVE_JOINT_NAMES,
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=0.0,
            velocity_limit_sim=SOLVER_VELOCITY_LIMIT,
            armature=0.0,
            friction=0.0,
        ),
    },
)
