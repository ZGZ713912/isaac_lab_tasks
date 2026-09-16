# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# deformable_V2 (狗v3) 资产配置 —— 变形底盘主动悬挂。
#
# USD：assets/usd_files/deformable_V2/deformable_V2.usd
#   （由 urdf/deformable_V2.urdf 经 Isaac Lab UrdfConverter 转换，
#     joint drive: force / target none；转换前置见 scripts/tools/
#     prepare_deformable_v2_urdf.py 与 convert_deformable_v2_urdf.sh）
#
# 16 DOF（URDF 顺序）：joint_leg_1..4 + joint_wheel_set_1..4 + joint_wheel_1..4
#                       + joint_upper_leg_1..4
#   - joint_leg_N ：平行四边形主动边（唯一驱动的腿关节）
#   - joint_wheel_set_N ：平四从动边；URDF <mimic> → PhysxMimicJointAPI(gearing=+1)，
#                         θ_ws ≡ +1·θ_leg（硬约束，由 PhysX 解算，env 不下发力矩）
#   - joint_upper_leg_N ：平四从动边；URDF <mimic> → gearing=−1，θ_upper ≡ −θ_leg
#   - joint_wheel_N ：全向轮，连续转动（URDF continuous，collision=解析球体 r=0.0769）
#
# 坐标约定：URDF 已在 prepare 阶段从 SolidWorks 的 Y-up 旋转为 RL 栈的 Z-up
#   （+X 车头、+Y 左、+Z 上），故模块内的初始位姿/接触/重力方向均按 Z-up 处理。
#
# 名义位姿（URDF 零位 q=0）：四轮心共面于 Z=-0.055，base 原点离地 ≈0.132 m。
#   注意 q 增大反而降低车体、增大轮距（0.428→0.585），与旧 deformable V1 相反。
#
# 所有执行器 stiffness/damping = 0（effort 模式）：env 只对 joint_leg_* 手工计算
# 位置 PD（kp=200/kd=4）并 set_joint_effort_target；此处只声明力矩/速度上限，
# 并显式覆盖 URDF 的占位值，确保不会被 PhysX 关节上限截断。
# wheel_set/upper_leg 不设驱动，仅由上述 mimic 硬约束跟随（力矩为约束反力）。
# =============================================================================

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from agent_world import AssetPath

"""Configuration for the deformable_V2 chassis (狗v3)."""

_LIMIT_LEG = 40.0  # N·m，与 env max_leg_torque / 部署 position_torque_max 同量级
_VEL_LEG = 17.0  # rad/s
_LIMIT_WHEEL = 5.0  # N·m
_VEL_WHEEL = 60.0  # rad/s

DeformableInfantryCFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/deformable_V2/deformable_V2.usd",
        activate_contact_sensors=True,
        copy_from_source=True,  # Required for proper articulation loading in Isaac Lab 2.3.0
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    # 名义站姿 q=0（base 离地 ~0.132）。spawn z=0.18 略高于接触高度，让车体轻微
    # 下落，避免初始过低导致轮/腿穿透地面；env 每次 reset 亦用 init_root_height。
    # joint_pos 用 ".*": 0.0：全部关节名义角为 0，且闭链自洽（ws=+1·0, upper=−1·0）。
    # 注意：不要写 "joint_wheel_.*" —— 它会同时匹配 joint_wheel_set_*，Isaac Lab 的
    # resolve_matching_names_values 会因多重匹配直接报错。
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.18),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 腿主动边：effort 模式，PD 在 env 内算
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["joint_leg_1", "joint_leg_2", "joint_leg_3", "joint_leg_4"],
            stiffness=0.0,
            damping=0.0,
            effort_limit=_LIMIT_LEG,
            velocity_limit=_VEL_LEG,
        ),
        # 平四从动边：由 <mimic> 硬约束跟随 leg（θ_ws=+1·θ_leg），不设驱动
        "wheel_set": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_wheel_set_1",
                "joint_wheel_set_2",
                "joint_wheel_set_3",
                "joint_wheel_set_4",
            ],
            stiffness=0.0,
            damping=0.0,
            effort_limit=_LIMIT_LEG,
            velocity_limit=_VEL_LEG,
        ),
        # 平四上连杆：由 <mimic> 硬约束跟随 leg（θ_upper=−1·θ_leg），不设驱动
        "upper_legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_upper_leg_1",
                "joint_upper_leg_2",
                "joint_upper_leg_3",
                "joint_upper_leg_4",
            ],
            stiffness=0.0,
            damping=0.0,
            effort_limit=_LIMIT_LEG,
            velocity_limit=_VEL_LEG,
        ),
        # 全向轮：连续转动（本任务零驱动，自由滚动）
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel_1", "joint_wheel_2", "joint_wheel_3", "joint_wheel_4"],
            stiffness=0.0,
            damping=0.2,
            effort_limit=_LIMIT_WHEEL,
            velocity_limit=_VEL_WHEEL,
            armature=0.0,
        ),
    },
)
