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

# ─────────────────────────────────────────────────────────────────────────────
# 【文件总览·给初学者】本文件是 V14 任务的"参数表"（纯配置，无运行逻辑）。
# @configclass 是 Isaac Lab 提供的装饰器，让普通 Python 类变成可嵌套/可覆盖的配置类。
# 组织方式 = 继承树：改一个任务只需继承基类并覆盖少数属性——
#   WheelbipeV14FlatEnvCfg（平地基类）
#     ├─ _v2          平地 + 云台航向锁定 + 小陀螺平移
#     │   └─ _v2_Play                上述任务的演示配置
#     ├─ _v1          平地 + 腾空落地预训练（从空中掉落学会落地）
#     │   └─ Rough_v1 ("跑场训练")   粗糙地形 + 跑场
#     ├─ Rough        ("小陀螺训练") 粗糙地形 + 旋转/平移
#     │   └─ Rough_Play              演示配置（带轨迹录制）
#     ├─ Flat_Play                   平地演示
#     ├─ FlatDreamWaq/_Play          DreamWaQ 算法变体（换观测空间）
#     ├─ FlatHIM/_Play               HIMLoco 算法变体
#     └─ FlatNP3OBarlow/_Play        NP3O 算法变体
# 文件里有大量被 # 注释掉的旧参数，属于"调参历史记录"，保留备用，不影响运行。
# 环境类（env.py）通过 self.cfg.xxx 读取这里的一切。
# ─────────────────────────────────────────────────────────────────────────────

from collections import OrderedDict  # 有序字典：保证 rewards 各项按书写顺序参与计算/展示
from dataclasses import field  # dataclass 工具：可变默认值(字典)要用 field(default_factory=...) 生成
import copy   # 深拷贝工具：配置对象互相继承时必须拷贝，避免几个任务共享同一份可变配置
import torch  # 只用到圆周率 torch.pi 和张量运算

import agent_tasks.manager.mdp.isaaclab as mdp  # 本项目的 mdp 库：奖励/事件/命令/地形等函数与配置都从这里取
import isaaclab.sim as sim_utils  # Isaac Lab 仿真工具：这里用到地面材质配置
from isaaclab.assets import ArticulationCfg  # 机器人"关节体"资产配置类型
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # 课程学习项配置
from isaaclab.managers import EventTermCfg as EventTerm  # 域随机化/事件项配置
from isaaclab.managers import SceneEntityCfg  # 指向场景里某个实体(机器人)+其关节/连杆子集的引用
from isaaclab.terrains import TerrainImporterCfg  # 地形导入配置（地形类型/生成器/材质）
from isaaclab.utils import configclass  # 配置类装饰器（支持默认值/继承/覆盖）
from isaaclab.utils.noise import NoiseModelCfg, UniformNoiseCfg  # 观测噪声模型配置（均匀噪声）
from agent_world.assets.wheelbipe_V14 import Wheelbipe_V14_CFG, Wheelbipe_V14_M3508_CFG, Wheelbipe_V14_No_Gimbal_CFG  # V14 一代机器人资产（带云台/带 M3508 电机/无云台 三种）
from agent_world.assets.wheelbipe_V14_2 import Wheelbipe_V14_2_CFG, Wheelbipe_V14_2_NG_CFG, Wheelbipe_V14_2_M3508_CFG  # V14 二代机器人资产（默认用这代）
from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env_cfg import EventCfg, Wheelbipe25v3FlatEnvCfg  # 父类：25 赛季 V3 的配置（大部分通用参数继承自它）
from agent_tasks.manager.mdp.terrain import TerrainCommandOverrideCfg  # "某地形上覆盖速度指令"的配置
from agent_tasks.direct.wheelbipe.wheelbipe_V14.cfg_utils import *  # V14 专用的常量/工具（观测裁剪表、重置姿势、课程默认参数等）


@configclass
class EventCfgV14(EventCfg):
    """Event configuration for the Wheelbipe V14 direct RL environments."""

    # ── 域随机化(Domain Randomization)事件表 ──
    # 目的：训练时故意随机化物理参数（质量/摩擦/增益…），让策略学到"对未知扰动鲁棒"，
    #      实机部署时才不会一碰真实世界就崩（sim2real 的核心手段）。
    # EventTerm 三要素：func=执行的随机化函数；mode=执行时机(startup=开局一次/reset=每次重置)；
    #                  params=传给函数的参数（范围写法 (min,max)）。
    # 继承自 25v3 的 EventCfg，这里覆盖 V14 关心的项；置 None 的项表示"禁用父类的该项"。

    add_base_mass = EventTerm(        # 车体质量随机化：×0.9~1.3（模拟负载变化）
        func=mdp.randomize_rigid_body_mass,
        mode="startup",               # 开局随机一次，整局保持
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),  # 只作用在 base_link 上
            # "mass_distribution_params": (1.0, 1.4),
            "mass_distribution_params": (0.9, 1.3),  # 质量缩放范围
            "operation": "scale",    # scale=按比例缩放
        },
    )
    # add_gimbal_mass = EventTerm(      # （注释掉的旧方案：云台质量随机化）
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["gimbal_yaw_link", "gimbal_pitch_link"]),
    #         "mass_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #     },
    # )
    base_inertia = EventTerm(         # 车体转动惯量随机化：×0.8~1.2
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    base_inertia = None               # ↑上面定义后立刻置 None = 实际禁用车体惯量随机化（惯量乱动容易学不稳）
    add_leg_mass = EventTerm(         # 腿部+云台各连杆质量随机化：×0.9~1.1
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(   # 覆盖所有腿连杆、弹簧连杆和云台
                "robot",
                body_names=[
                    ".*_front1_link",   # 前腿各节
                    ".*_front2_link",
                    ".*_front3_link",
                    ".*_front4_link",
                    ".*_rear1_link",    # 后腿各节
                    ".*_rear2_link",
                    ".*_spring1_link",  # 弹簧连杆
                    ".*_spring2_link",
                    "gimbal_yaw_link",  # 云台
                    "gimbal_pitch_link",
                ],
            ),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    # add_leg_mass = None
    add_wheel_mass = EventTerm(       # 轮子质量随机化：×0.9~1.1
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    wheels_inertia = EventTerm(       # 轮子转动惯量随机化：×0.8~1.2（影响轮子加减速响应）
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    wheels_inertia = None             # 同上：定义后禁用
    # gimbal_com = EventTerm(           # （注释掉：云台质心随机化）
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["gimbal_yaw_link", "gimbal_pitch_link"]),
    #         "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
    #     },
    # )
    # gimbal_inertia = EventTerm(       # （注释掉：云台惯量随机化）
    #     func=mdp.randomize_rigid_body_inertia,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["gimbal_yaw_link", "gimbal_pitch_link"]),
    #         "inertia_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )
    # gimbal_inertia = None
    base_com = EventTerm(             # 车体质心随机化：x ±4cm、y/z ±2cm（模拟载重偏置）
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.04, 0.04), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    base_material = EventTerm(        # 车体摩擦材质随机化：摩擦故意很低(0.01~0.1)，模拟车壳光滑
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "static_friction_range": (0.01, 0.1),    # 静摩擦范围
            "dynamic_friction_range": (0.01, 0.1),   # 动摩擦范围
            "restitution_range": (0.02, 0.2),        # 弹性(恢复系数)范围
            "num_buckets": 64,       # 材质分桶数（物理引擎按桶批量算）
            "make_consistent": True, # 保证静摩擦≥动摩擦
        },
    )
    guide_material = EventTerm(       # guide 机构摩擦随机化（V14 无 guide，实际不生效）
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_guide_link"),
            "static_friction_range": (0.1, 0.7),
            "dynamic_friction_range": (0.1, 0.7),
            "restitution_range": (0.01, 0.1),
            "num_buckets": 8,
            "make_consistent": True,
        },
    )
    physics_material = EventTerm(     # ★轮胎摩擦随机化：静摩擦 0.5~1.2（影响打滑程度，最重要）
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.02, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    # legs_act_joint_frictions = EventTerm(  # （注释掉：主动腿关节摩擦的旧参数）
    #     func=mdp.randomize_joint_parameters_v1,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_rear1_joint", ".*_front1_joint"]),
    #         "static_friction_distribution_params": (0.25, 1.0),
    #         "dynamic_friction_distribution_params": (0.15, 0.6),
    #         "viscous_friction_distribution_params": (0.05, 0.25),
    #         # "armatuleft_wheel_link material=static=1.1905 dynamic=0.9076 restitution=0.1467 contact=|F|=10.43 peak=33.36re_distribution_params": (0.001, 0.003),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )
    leg_front_joint_frictions = EventTerm(  # 前腿主动关节摩擦随机化：在默认值上"加"随机量
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_front1_joint"]),  # 4 个前腿髋关节
            # "static_friction_distribution_params": (2.0, 4.0),
            # "dynamic_friction_distribution_params": (1.0, 2.0),
            # "viscous_friction_distribution_params": (0.02, 0.1),
            # "armature_distribution_params": (0.001, 0.003),
            # "static_friction_distribution_params": (1.5, 2.0),
            # "dynamic_friction_distribution_params": (1.4, 1.8),
            # "viscous_friction_distribution_params": (0.01, 0.1),
            "static_friction_distribution_params": (0.25, 1.0),   # 库仑静摩擦附加量
            "dynamic_friction_distribution_params": (0.25, 1.0),  # 库仑动摩擦附加量
            "viscous_friction_distribution_params": (0.05, 0.2),  # 粘性摩擦附加量(与速度成正比)
            "operation": "add",      # add=在模型默认值上叠加
            "distribution": "uniform",  # 均匀分布采样
        },
    )
    leg_rear_joint_frictions = EventTerm(   # 后腿主动关节摩擦随机化（参数同前腿）
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_rear1_joint"]),  # 4 个后腿髋关节
            # "static_friction_distribution_params": (1.5, 3.0),
            # "dynamic_friction_distribution_params": (0.75, 1.5),
            # "viscous_friction_distribution_params": (0.01, 0.1),
            # "armature_distribution_params": (0.001, 0.003),
            # "static_friction_distribution_params": (0.9, 1.4),
            # "dynamic_friction_distribution_params": (0.8, 1.2),
            # "viscous_friction_distribution_params": (0.01, 0.1),
            "static_friction_distribution_params": (0.25, 1.0),
            "dynamic_friction_distribution_params": (0.25, 1.0),
            "viscous_friction_distribution_params": (0.05, 0.2),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    wheel_joint_frictions = EventTerm(      # 轮关节摩擦随机化：静摩擦 0.05~0.25、粘性 0~0.01（轮轴承阻力）
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"),
            "static_friction_distribution_params": (0.05, 0.25),
            # "static_friction_distribution_params": (0.025, 0.125),
            # "dynamic_friction_distribution_params": (0.025, 0.125),
            # "dynamic_friction_distribution_params": (0.05, 0.25),
            "viscous_friction_distribution_params": (0.0, 0.01),
            # "viscous_friction_distribution_params": (0.0, 0.005),
            # "armature_distribution_params": (0.00, 0.003),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # wheel_joint_frictions = None
    legs_inact_joint_frictions = EventTerm( # 腿部从动关节（无电机的那几节）摩擦随机化
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(    # rear2/front2~4/spring1 这些被动关节
                "robot",
                joint_names=[
                    ".*_rear2_joint",
                    ".*_front2_joint",
                    ".*_front3_joint",
                    ".*_front4_joint",
                    ".*_spring1_joint",
                ],
            ),
            "static_friction_distribution_params": (0.05, 0.1),  # 从动件摩擦较小
            # "dynamic_friction_distribution_params": (0.025, 0.05),
            # "dynamic_friction_distribution_params": (0.05, 0.1),
            "viscous_friction_distribution_params": (0.01, 0.025),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # spring_frictions = EventTerm(     # （注释掉：弹簧关节摩擦随机化）
    #     func=mdp.randomize_joint_parameters_v1,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_spring2_joint"),
    #         "static_friction_distribution_params": (0.1, 1.0),
    #         "viscous_friction_distribution_params": (25., 75.),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )
    gimbal_joint_frictions = EventTerm(     # 云台两关节摩擦随机化（很小：真实云台轴承很顺滑）
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gimbal_yaw_joint", "gimbal_pitch_joint"]),
            "static_friction_distribution_params": (0.002, 0.01),
            # "dynamic_friction_distribution_params": (0.001, 0.005),
            # "dynamic_friction_distribution_params": (0.002, 0.01),
            "viscous_friction_distribution_params": (0.002, 0.01),
            # "armature_distribution_params": (0.0, 0.002),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(  # ★全部执行器 PD 增益随机化：×0.75~1.25
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,  # 同一环境两次触发至少隔 720 步（约 14 秒换一副增益）
        mode="reset",            # 每次环境重置时重新随机
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # 所有关节
            # "stiffness_distribution_params": (0.75, 1.5),
            # "damping_distribution_params": (0.75, 1.5),
            "stiffness_distribution_params": (0.75, 1.25),  # 刚度缩放范围
            "damping_distribution_params": (0.75, 1.25),    # 阻尼缩放范围
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    spring_damping = EventTerm(       # 弹簧关节的刚度/阻尼单独随机化（弹簧软硬影响被动减震）
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_spring2_joint"),  # 弹簧关节
            "stiffness_distribution_params": (0.01, 0.01),  # 刚度基本不动
            # "damping_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.5, 1.5),      # 阻尼 ×0.5~1.5（模拟减震器差异）
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    leg_effort_noise = EventTerm(     # 腿电机输出力矩扰动（默认范围不改变行为，训练时可放开）
        func=mdp.randomize_actuator_effort_output,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_front1_joint",".*_rear1_joint"]),  # 8 个腿电机
            # 输出力矩扰动：tau = clip(tau_nominal * scale + bias + N(0, noise_std))
            # 默认配置不改变行为，需要训练时可把 scale/bias/noise_std 的范围放开。
            "effort_scale_distribution_params": (0.8, 1.1),   # 力矩整体缩放
            "effort_bias_distribution_params": (0.0, 0.0),    # 固定偏置（当前关）
            "effort_noise_std_distribution_params": (0.0, 0.0),  # 随机噪声幅度（当前关）
            "distribution": "uniform",
        },
    )
    wheel_effort_noise = EventTerm(   # 轮电机输出力矩扰动（同上）
        func=mdp.randomize_actuator_effort_output,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"),  # 2 个轮电机
            # 输出力矩扰动：tau = clip(tau_nominal * scale + bias + N(0, noise_std))
            # 默认配置不改变行为，需要训练时可把 scale/bias/noise_std 的范围放开。
            "effort_scale_distribution_params": (0.9, 1.1),
            "effort_bias_distribution_params": (0.0, 0.0),
            "effort_noise_std_distribution_params": (0.0, 0.0),
            "distribution": "uniform",
        },
    )
    reset_base = EventTerm(           # 重置时车体初始姿态随机化：roll/pitch ±0.15rad、yaw 全向
        func=mdp.reset_root_state_uniform_vel_b,
        mode="reset",
        params={
            "pose_range": {           # 姿态扰动范围（弧度）
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-3.14, 3.14), # 朝向随机（各方向都要会走）
            },
            "velocity_range": {},     # 初速度不加扰动
        },
    )
    # base_external_force_torque_xyz = None


@configclass
class EventCfgV14_Play(EventCfgV14):
    """Play-mode event configuration for Wheelbipe V14."""

    # Play(演示)模式的事件表：完整继承训练版的域随机化定义，
    # 但父类环境的 play 流程会把 startup/reset 类随机化弱化或跳过，演示更贴近真机。




@configclass
class CurriculumCfgV14:
    """Wheelbipe V14 command curriculum configuration."""

    # ── 课程学习配置 ──
    # 课程学习(curriculum)：不一开始就上最难的目标，而是按训练表现逐步加码。
    # CurrTerm 的 func 会在训练过程中被调用，根据指标自动调整环境参数。

    track_height_progression = CurrTerm(   # 课程项1：按"身高追踪"表现分阶段调奖励权重
        func=mdp.RewardWeightProgression,  # 通用函数：监控某奖励项的表现，达标后进入下一阶段
        params={ 
            "reward_key": "track_height_exp",      # 监控的奖励项：身高追踪
            "num_steps_per_env": 24,               # 每轮迭代的步数（和 PPO 一致）
            "window_size": 64,                     # 用最近 64 轮的平均表现判断是否达标
            "min_stage_episodes": 64,              # 每阶段至少跑满 64 局才允许晋级
            "normalize_by_episode_length": True,   # 按局长归一化奖励再统计
            # 最后一阶段再次达标后恢复默认 reward，即 reward_scale 回到 1.0。
            "restore_defaults_on_last_stage_threshold": True,
            "stages": [                            # 阶段列表：每阶段定义权重/门槛/最少局数
                {
                    "reward_weights": {            # 第 1 阶段（入门）：提高身高追踪权重
                        # "flat_orientation_y": -1.0,
                        # "flat_orientation_y_v": -1.0,
                        # "flat_orientation_x": -1.0,
                        # "flat_orientation_x_v": -1.0,
                        "track_height_exp": 1.0,           # 身高追踪 exp 奖励权重 1.0
                        "track_height_exp_tight": 1.0,     # 严格版身高追踪权重 1.0
                        # "track_height_exp_soft": 2.0,
                        # "track_height_exp_both_wheels_contact": 5.0,
                        # "lin_vel_z": -0.1,
                        # "rear2_rear1_joint_pos_limits": -1.0,
                        # "rear2_rear1_joint_pos_limits_torque": -1.0,
                        # "rear2_rear1_joint_pos_limits_vel": -1.0,
                        # "termination": -100.0,
                    },
                    "reward_scale": {              # 可同时缩放其它奖励（平滑项），当前未启用
                        # "action_smoothness_leg": 0.1,
                        # "leg_joint_acc": 0.1,
                        # "action_rate": 0.1,


                    },
                    "threshold": 0.4,              # 达标线：该奖励表现 ≥0.4 才晋级
                    "min_episodes": 500,           # 且本阶段至少经历 500 局
                },
                {
                    "reward_weights": {            # 第 2 阶段（进阶）：稍微降低追踪权重（防过拟合单一目标）
                        # "flat_orientation_y": -1.0,
                        # "flat_orientation_y_v": -1.0,
                        # "flat_orientation_x": -1.0,
                        # "flat_orientation_x_v": -1.0,
                        "track_height_exp": 0.8,
                        "track_height_exp_tight": 0.6,
                        # "track_height_exp_soft": 0.5,
                        # "track_height_exp_both_wheels_contact": 1.0,
                        # "lin_vel_z": -0.5,
                        # "termination": -100.0,
                        # "termination": -10.0,
                    },
                    "reward_scale": {
                        # "action_smoothness_leg": 0.5,
                        # "leg_joint_acc": 0.5,
                        # "action_rate": 0.5,
                    },
                    "threshold": 0.4,
                    "min_episodes": 500,
                },
                # {
                #     "reward_weights": {            # （注释掉的第 3 阶段）
                #         "track_height_exp": 0.4,
                #         "track_height_exp_tight": 1.0,
                #         "lin_vel_z": -1.0,
                #     },
                # },
            ],
        },
    )

    base_vertical_assist_force_progression = CurrTerm(  # 课程项2：给车体一个向上的"助力托举力"
        func=mdp.BaseVerticalAssistForceProgression,    # 按身高追踪表现逐步撤掉助力
        params={
            "reward_key": "track_height_exp",           # 同样以身高追踪表现为准
            "num_steps_per_env": 24,
            "window_size": 64,
            "min_stage_episodes": 64,
            "normalize_by_episode_length": True,
            "apply_on_compute": True,                   # 在奖励计算前施加力
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),  # 力作用在车体上
            "stages": [                                 # 三阶段：托举力 160N → 80N → 0N
                {                                       # 初期像"扶着学步车"，让机器人先学会平衡
                    "force_z": 160.0,
                    "threshold": 0.4,
                    "min_episodes": 500,
                },
                {
                    "force_z": 80.0,                    # 减半
                    "threshold": 0.4,
                    "min_episodes": 500,
                },
                {
                    "force_z": 0.0,                     # 最终完全靠自己站立
                },
            ],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# V14 平地基类：所有 V14 任务的"默认参数"都在这里，其它任务类继承后只改差异项。
# 属性分几大类：机器人与执行器 / 观测（噪声、延迟、维度）/ 动作延迟 /
# 弹簧悬挂 / 高度指令 / 状态机 / 奖励权重表(rewards) / __post_init__ 里的运行时调整。
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class WheelbipeV14FlatEnvCfg(Wheelbipe25v3FlatEnvCfg):
    """Configuration for the Wheelbipe V14 direct RL environment with flat terrain."""

    # PPO runner 每轮采集步数，用于从 common_step_counter 外推训练 iteration。
    training_progress_steps_per_iteration = 24  # PPO 每轮每环境采 24 步（num_steps_per_env，见 agents 配置）

    # Temporarily disable domain randomization for V14 training.
    events = EventCfgV14()            # 换用上面定义的 V14 域随机化事件表
    # curriculum = CurriculumCfgV14()
    curriculum = None                 # 课程学习关闭（None = 用不到课程；需要时换回上面的 CurriculumCfgV14()）
    play_keep_done_reset = True       # Play 模式下"到时重置"照常执行（保持演示节奏）
    # reset_heading_axis_aligned_only = True
    robot_cfg: ArticulationCfg = Wheelbipe_V14_2_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()  # ★用哪台机器人：V14 二代，挂到每个环境自己的路径下
    # robot_cfg: ArticulationCfg = Wheelbipe_V14_No_Gimbal_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()  # （备选：无云台版）
    # robot_cfg = Wheelbipe_V14_2_NG_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()  # （备选：二代无云台）

    # —— 从执行器配置里取出各组关节名的正则表达式（父类靠它们识别关节用途）——
    legs_act_name = robot_cfg.actuators["legs_act"].joint_names_expr   # 主动腿关节（4 个髋电机）
    legs_inact_name = robot_cfg.actuators["legs_inact"].joint_names_expr  # 从动腿关节（无电机）
    wheel_name = robot_cfg.actuators["wheel"].joint_names_expr         # 轮电机（2 个）
    spring_name = robot_cfg.actuators["spring"].joint_names_expr       # 弹簧关节（2 个）
    gimbal_yaw_name = robot_cfg.actuators["gimbal_yaw"].joint_names_expr    # 云台 yaw 关节
    gimbal_pitch_name = robot_cfg.actuators["gimbal_pitch"].joint_names_expr  # 云台 pitch 关节
    use_gimbal = True                 # 启用云台
    # gimbal_yaw_name = None
    # gimbal_pitch_name = None

    gimbal_pitch_target_pos: float = -0.5           # 云台 pitch 固定目标角（-0.5 rad ≈ 抬头）
    gimbal_yaw_velocity_range: tuple[float, float] = (-torch.pi, torch.pi)  # yaw 自旋速度采样范围 ±π rad/s
    ordered_leg_joint_names = V14_ORDERED_LEG_JOINT_NAMES  # 运动学要求的 12 关节固定顺序（左右对称成对）
    ordered_leg_body_names = V14_ORDERED_LEG_BODY_NAMES    # 对应的 12 条连杆固定顺序
    # links_length = V14_LINKS_LENGTH
    # alpha_offset = V14_ALPHA_OFFSET

    mute_wheel_pos_obs = True        # 不把轮子角度放进观测：轮子可无限旋转，位置没有信息量
    default_height_cmd = 0.22        # 默认身高指令 0.22 m（轮轴到底盘的伸缩高度）

    # —— 观测延迟模拟（sim2real：真机传感/通信有延迟，训练时就让策略适应）——
    # 每项 = [最小延迟步数, 最大延迟步数]，重置时在区间内随机取固定值
    obs_delay_cfg = {
        "root_ang_vel_b": [1, 4],        # 角速度观测延迟 1~4 步
        "projected_gravity_b": [1, 4],   # 重力方向(姿态)延迟
        "joint_pos": [1, 4],             # 关节角延迟
        "joint_vel": [1, 4],             # 关节速度延迟
    }
    # obs_delay_cfg = {
    #     "root_ang_vel_b": [2, 5],
    #     "projected_gravity_b": [2, 5],
    #     "joint_pos": [2, 5],
    #     "joint_vel": [2, 5],
    # }
    obs_history_len = 10             # 观测历史堆叠 10 帧（可从短历史推断速度/趋势）
    obs_default_time_lag = 1         # 默认时间错位 1 步
    use_obs_delay = True             # 开启观测延迟

    # —— 动作延迟模拟（真机电机响应滞后）——
    act_delay_cfg = {
        "leg_actions": [1, 3],           # 腿动作延迟 1~3 步
        "wheel_actions": [1, 3],         # 轮动作延迟 1~3 步
    }
    # act_delay_cfg = {
    #     "leg_actions": [1, 4],
    #     "wheel_actions": [1, 4],
    # }
    use_act_delay = True             # 开启动作延迟

    # Temporarily disable observation noise for V14 training.
    ''' noise '''
    # —— 观测噪声：给每个观测量加均匀噪声（模拟传感器精度），提高鲁棒性 ——
    self_obs_noise_cfg = {
        'root_ang_vel_b': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.25, n_max=0.25)),    # 角速度 ±0.25
        'projected_gravity_b': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),  # 姿态(重力方向) ±0.05
        'joint_pos': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.025, n_max=0.025)),       # 关节角 ±0.025 rad
        'leg_joint_vel': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.5, n_max=0.5)),       # 腿关节速度 ±0.5
        'wheel_joint_vel': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-1.0, n_max=1.0)),     # 轮速 ±1.0
        'joint_torque': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.25, n_max=0.25)),      # 力矩 ±0.25
        'lin_vel': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),             # 线速度 ±0.1
        'height': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01)),            # 高度 ±1cm
    }

    # self_act_noise_cfg = {
    #     'leg_actions': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.025, n_max=0.025)),
    #     'wheel_actions': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.5, n_max=0.5)),
    # }
    self_act_noise_cfg = None        # 动作噪声关闭（噪声已经够多了）

    # —— 弹簧悬挂模型参数（腿末端的被动减震）——
    spring_settings = dict(
        mode = 'linear', # 'constant','linear','curve'  # 弹簧力模型：线性(力随压缩行程增大)
        constant_force = 240., # 固定弹簧力                  # 常力模式的恒定力 240N（当前用线性，此项备用）
        random_force = [-50.,50.], # 随机弹簧力范围           # 弹簧力随机扰动 ±50N（域随机化）
        # spring_offset = 0.03876, # 初始位置弹簧压缩行程m
        spring_offset = 0.06076,     # 初始压缩行程 6cm
        linear_up = 600.*1.0, # 线性模式下最大压缩弹簧力N     # 压到底时的最大弹力 600N
        linear_down = 400.*1.0, # 线性模式下最小压缩弹簧力N    # 刚开始压缩的弹力 400N
        linear_length = 0.07, # 线性模式下弹簧力从最小到最大变化的行程m  # 7cm 行程内力从 400 线性升到 600
        damping = False,           # 是否启用阻尼模型
        rand_stretch_damping_range = [300,800],   # 拉伸方向阻尼随机范围
        rand_contract_damping_range = [300,800],  # 压缩方向阻尼随机范围
    )

    use_leg_length_as_height = False # 是否用"腿长"代替"车体高度"作为高度指令（False=用车体离地高）
    height_range = [0.20, 0.42]      # 身高指令采样范围 0.20~0.42 m
    terrain_command_overrides: dict[str, TerrainCommandOverrideCfg] = field(default_factory=dict)  # 按地形名覆盖速度指令的表（粗糙任务里填）
    terrain_command_switch_hold_steps: int = 0       # 地形命令切换后保持的步数
    use_absolute_height = True       # 用绝对高度观测（无需地面估计/高度扫描仪，省算力）
    use_wheel_vel_control = True     # 轮子用速度控制（而非力矩控制）
    front_rear_joint_limit_rewards_enabled = False   # 前后关节限位奖励开关
    rear2_rear1_joint_limit_lower = -1.0 / 180.0 * torch.pi   # rear2-rear1 关节角下限 -1°
    rear2_rear1_joint_limit_upper = 68.5 / 180.0 * torch.pi   # 上限 68.5°
    rear2_rear1_joint_limit_boundary_ratio = 0.03    # 限位边界缓冲比例 3%
    rear2_rear1_joint_limit_lower_boundary_ratio = 0.03
    rear2_rear1_joint_limit_upper_boundary_ratio = 0.03
    rear2_rear1_joint_limit_vel_threshold = 10.0     # 限位处速度阈值（防撞限位）

    # —— 五连杆腿部运动学几何参数（由关节角算腿长/轮位置用）——
    links_length = [0.1134,0.135,0.210]   # 三段连杆长度(米)
    alpha_offset = [-6.61/180.*torch.pi,  # 各连杆的安装角偏移(弧度)
                    torch.pi,
                    29.7/180.*torch.pi,
                    (180.-6.61-2*29.7)/180.*torch.pi,
                    (29.7)/180.*torch.pi,
                    29.7/180.*torch.pi]
    leg_length_range = [0.13,0.32]        # 腿长采样范围(米)
    leg_angle_range = [-0.5*torch.pi,0.75*torch.pi]  # 腿摆角采样范围
    use_predefined_leg_random_start = True   # 重置时用"预定义腿部姿势"随机挑选（见下）
    predefined_reset_ground = copy.deepcopy(V14_PREDEFINED_RESET_GROUND)  # 预定义的地面初始姿势集合（蹲/站等）
    enable_state_machines = False         # 状态机总开关（腾空/跳台阶等，V14 基础版关闭）
    termination_duration_enabled = True   # 终止判定需要"连续多步满足"才生效
    termination_duration_steps = 20       # 连续 20 步(0.4s)都异常才算摔倒（防瞬时误判）
    reset_heading_target_terminate_enabled = False   # 朝向偏离是否终止（关闭）
    reset_heading_target_terminate_threshold_deg = 20.0
    # —— 控制模式观测：7 维槽位告诉策略"当前模式/指令"（平移/爬梯/跳跃/身高目标等）——
    ctrl_mode_obs_enabled = True          # 启用该 7 维额外观测
    ctrl_mode_obs_dim = 7                 # 维度 7
    ctrl_mode_obs_layout = (              # 每一维的含义（普通模式布局）
        "normal",        # 0: 普通模式标志
        "stair",         # 1: 爬梯模式
        "slope",         # 2: 坡道模式
        "recover",       # 3: 恢复模式
        "jump",          # 4: 跳跃模式
        "height_target", # 5: 目标高度
        "state_time",    # 6: 模式已持续步数(归一化)
    )
    # Backward-compatible aliases. Prefer ctrl_mode_obs_* for new configs.
    # 旧名字的别名（兼容历史代码，新配置一律用 ctrl_mode_obs_*）
    jump_takeoff_extra_obs_enabled = ctrl_mode_obs_enabled
    jump_takeoff_extra_obs_dim = ctrl_mode_obs_dim
    jump_takeoff_extra_obs_layout = ctrl_mode_obs_layout
    jump_takeoff_extra_obs_slope_terrain_names = (   # 哪些坡类地形会触发 slope 模式标志
        "slope_for_rm_low",
        "slope_for_rm_high",
        "inv_slope_for_rm_low",
        "inv_slope_for_rm_high",
        "stair_slope_for_rm_low",
        "stair_slope_for_rm_high",
        "inv_stair_slope_for_rm_low",
        "inv_stair_slope_for_rm_high",
        "cliff_inv_stair_slope_for_rm",
        "cliff_inv_stair_slope_tall_for_rm",
    )

    # —— 腾空-落地状态机配置（大字典：进入/退出条件、落地轨迹、专属奖励等）——
    airborne_state_machine_cfg = {
        "enabled": False,                       # 基础平地任务关闭（_v1 腾空预训练任务里打开）
        "allowed_terrain_names": (),            # 限定哪些地形允许进入腾空（空=不限）
        "not_allowed_terrain_names": (),        # 禁止进入腾空的地形
        "enter": {                              # 判定"腾空"的条件
            "wheel_radius": 0.06,               # 轮半径 6cm
            "body_height_threshold": 0.3,       # 车身高于 0.3m 视为可能腾空
            "wheel_clearance_threshold": 0.08,  # 轮离地 >8cm 视为腾空
            "duration_s": 0.02,                 # 持续 0.02s 才确认
        },
        "target_height": {                      # 腾空时是否修改身高目标
            # 关闭后 airborne 状态机不再修改高度奖励的目标高度，
            # 也不再改写高度奖励使用的参考高度计算口径。
            "enabled": False,
            "bias": 0.12,                       # 若开：目标身高加 0.12m 偏置
            "max": 0.36,                        # 上限 0.36m
        },
        "landing_trajectory": {                 # 落地缓冲轨迹：检测到快落地时规划一条二次曲线让腿缓冲
            "enabled": False,
            # 任一轮接触计时达到该阈值后，若还在下落，则初始化一次落地缓冲轨迹。
            "start_wheel_contact_duration_s": 0.02,   # 轮子接触计时阈值
            # 固定时长结束时的参考高度；若触发起点低于 target_height + min_height_margin，则不规划。
            "target_height": 0.22,              # 轨迹终点高度
            # 固定时长结束时的参考 z 速度，默认 0，且不会允许配置为正值。
            "end_vel_z": 0.0,                   # 轨迹终点竖直速度（落地时归零）
            "min_height_margin": 0.02,
            # 只有 z 方向速度小于 -min_down_vel 时才触发，避免轻微噪声启动轨迹。
            "min_down_vel": 0.2,                # 下落速度 >0.2m/s 才触发
            # 固定二次轨迹总时长；到 duration_s 时刚好到达终点高度和终点速度。
            "duration_s": 0.3,                  # 轨迹总时长 0.3s
            # 可选加速度上限；若二次轨迹所需常加速度超过该值，则不启动本次轨迹。
            "max_abs_acc": 30.0,                # 轨迹加速度上限 30 m/s²
        },
        "exit": {                               # 判定"落地结束"的条件
            "wheel_contact_force_threshold": 20.0,      # 轮子接触力阈值 20N
            "wheel_contact_height_threshold": 0.15,     # 轮高阈值
            "wheel_contact_duration_s": 0.3,            # 轮子持续触地 0.3s
            "base_contact_force_threshold": 5.0,        # 车体触地力阈值（车体触地=摔了）
            "base_contact_duration_s": 0.25,
            "max_duration_s": 1.,                       # 腾空状态最长 1s
        },
        "reward_scales": {                      # 腾空状态下各奖励项的权重覆盖
            "undesired_contact": 25.0,          # 腾空期间乱接触罚得更重(×25)
            # "flat_orientation_y_v": 2.0,
            # "flat_orientation_x_v": 2.0,
            "foot_bound_square": 1.0,           # 脚(轮)乱甩惩罚
            # "track_height_exp_tight": 1.0,
            # "track_height_square": 0.0,
            "termination": 3,                   # 摔倒罚分调整
        },
        "reward_full": {                        # 腾空期间的"满额奖励"覆盖表（当前空）
        },
        "reward_additions": {                   # 腾空专属的附加奖励项定义（当前空，_v1 里填充）
        },
        "terrain_command_resample": {           # 腾空期间重采速度指令（模拟跳向不同方向）
            "enabled": False,
            # 进入 airborne 时按概率决定是否启用本次 airborne 临时速度命令覆盖。
            # profiles 的 key 默认就是地形名；也可在 profile 内用 terrain_names 指定多个地形。
            "prob": 0.15,                       # 15% 概率触发
            # True 时，lin_vel_x 重采样符号跟当前目标速度保持一致：
            # 当前目标为正则只采样 >=0，为负则只采样 <=0。
            "lin_vel_x_sign_from_current": True,
            "profiles": {                       # 各地形的临时指令范围
                "high_stair_for_rm": {
                    "lin_vel_x": [(-1.5, 1.5)],     # x 速度 ±1.5
                    # "lin_vel_x": [(-0., 0.)],
                    "lin_vel_y": (0.0, 0.0),        # y 速度 0
                    "ang_vel_z": (-1.0, 1.0),       # 偏航角速度 ±1
                },
                "high_speed_stair_for_rm": {    # 高速台阶地形（范围同上）
                    "lin_vel_x": [(-1.5, 1.5)],
                    # "lin_vel_x": [(-0., 0.)],
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-1.0, 1.0),
                },
                "low_speed_stair_for_rm": {     # 低速台阶地形（范围同上）
                    "lin_vel_x": [(-1.5, 1.5)],
                    # "lin_vel_x": [(-0., 0.)],
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-1.0, 1.0),
                },
            },
        },
    }
    jump_takeoff_permission_cfg = {         # 跳跃"许可"层：决定哪些环境被允许触发跳跃
        # 独立于 special_modes 的跳跃许可层。采样到许可的 env 才允许
        # jump_takeoff_state_machine 的 random/flag/manual trigger 生效。
        "enabled": False,
        # 每次 command resample 时，对所有 env 独立采样许可的比例。
        # 它不占用 special_mode 桶，因此可以和 spin/dash 等模式同时存在。
        "rel_envs": 0.0,                    # 允许跳跃的环境比例
        "iteration_start": 0,               # 从第几轮开始生效
        "iteration_end": -1,                # -1 = 永不过期
        "steps_per_iteration": 24,
        # None 表示只提供跳跃许可，不覆盖速度命令。
        # 配置后优先级高于 normal/special_mode，低于 terrain command override。
        "ranges": None,                     # 可选：跳跃时的速度指令覆盖
        # None 表示不覆盖 height_cmd。配置后同样低于 terrain height override。
        "height_range": None,               # 可选：跳跃时的高度指令覆盖
    }
    wheel_forward_scan_cfg = {              # 轮前地形扫描：往前看一步，检测到台阶就自动抬高身高
        "enabled": False,
        "scan": {
            "forward_offset": 0.5,          # 在轮前 0.5m 处采样地形高度
        },
        "detect": {
            "step_height_min": 0.12,        # 台阶高度下限 12cm
            "step_height_max": 0.14,        # 上限 14cm
            "wall_height": 0.14,            # 高于 14cm 视为墙
        },
        "height_cmd": {
            "bias": 0.16,                   # 检测到台阶时身高指令 +0.16m
            "hold_s": 2.0,                  # 保持 2 秒
            "max": 0.40,                    # 身高上限 0.40m
        },
    }
    undesired_contact_force_threshold = 3.0   # "乱接触"的判定力阈值 3N
    desired_contact_force_threshold = 5.0     # "正常接触"(轮子)的判定阈值 5N

    # —— 观测堆叠（帧叠加）——
    use_frame_stack = False           # 不做帧堆叠（基础版用单帧+历史由算法侧管理）
    num_obs_hist = 1                  # 策略观测历史长度 1
    num_privileged_obs_hist = 1       # critic 特权观测历史长度 1

    # —— 速度奖励的"门控"：姿态/身高太差时关闭速度奖励（先站稳再谈跑）——
    vel_upright_gate_enabled: bool = False          # 门控总开关（各任务可单独开）
    vel_upright_gate_sigma: float = 0.1             # 站姿门控的平滑度
    vel_orientation_y_gate_enabled: bool = False    # 俯仰门控
    vel_orientation_y_gate_full_deg: float = 5.0    # 偏离 5° 内奖励全额
    vel_orientation_y_gate_zero_deg: float = 20.0   # 偏离 20° 以上奖励归零
    vel_height_gate_enabled: bool = False           # 身高门控
    vel_height_gate_mode: str = "linear_band"       # 线性带状衰减
    vel_height_gate_full_error: float = 0.05        # 误差 5cm 内全额
    vel_height_gate_zero_error: float = 0.1         # 误差 10cm 归零
    vel_height_gate_tracker_sigma: float = 0.02
    height_upright_gate_enabled: bool = False       # 身高奖励的站姿门控
    height_upright_gate_sigma: float = 0.1
    stand_still_deadzone_enabled: bool = True       # "站住"死区：指令速度≈0 时按站住判定
    stand_still_deadzone_threshold: float = 0.1     # 死区阈值 0.1 m/s
    # —— 轮电机轴对齐奖励参数（保持轮轴水平=身体不歪）——
    wheel_motor_z_axis_align_ref_y_offset: float = 0.20855  # 参考点 y 偏移
    wheel_motor_z_axis_align_tolerance: float = 0.0         # 容差
    wheel_motor_z_axis_align_sigma: float = 0.01            # 对齐奖励 σ
    wheel_motor_z_axis_align_tight_sigma: float = 0.001     # 严格版 σ
    play_wheel_motor_z_axis_align_debug: bool = False       # Play 调试打印开关
    play_wheel_motor_z_axis_align_debug_interval: int = 50  # 打印间隔(步)
    play_wheel_motor_z_axis_align_debug_env_id: int = 0     # 打印哪个环境
    play_wheel_material_debug: bool = True                  # Play 时打印轮子摩擦参数
    play_wheel_material_debug_interval: int = 50
    play_wheel_material_debug_env_id: int = 0

    # —— 数值诊断调试：训练时检测异常大的状态/奖励并打印 top-k ——
    debug_value_diagnosis: bool = False
    debug_value_diagnosis_interval: int = 50
    debug_value_diagnosis_topk: int = 3
    debug_value_diagnosis_threshold_only: bool = False
    debug_value_diagnosis_thresholds: dict = {   # 各量的"异常"报警阈值
        "reward_total_abs": 1.0,                 # 总奖励绝对值
        "reward_term_abs": 0.5,                  # 单项奖励绝对值
        "state_joint_vel_wheel_abs": 120.0,      # 轮速
        "state_root_lin_vel_b_abs": 20.0,        # 线速度
        "state_root_ang_vel_b_abs": 20.0,        # 角速度
        "state_applied_torque_abs": 350.0,       # 力矩
        "obs_policy_abs": 100.0,                 # 策略观测
        "obs_critic_abs": 100.0,                 # critic 观测
    }
    obs_input_clip_cfg: dict = V14_BASIC_OBS_CLIP   # 观测裁剪表：把异常观测夹回正常范围（防网络被极端值带偏）
    # Promote the stable observation scaling from experiment 001 into the default
    # V14 flat task so the policy is not dominated by high-magnitude velocity terms.
    obs_input_scale_enabled: bool = True         # 观测缩放开关：把各观测量缩到相近量级（轮速很大不 dominate）
    obs_input_scale_streams: tuple[str, ...] = ("policy","critic")  # 对策略和 critic 两组观测都缩放
    obs_input_scale_cfg: dict = V14_BASIC_OBS_SCALE  # 缩放系数表（每维一个系数）
    joint_pos_obs_encoding: str = "raw"          # 关节角观测编码方式：raw=直接用弧度值
    # —— critic 特权观测：critic 训练时能"作弊"看到策略看不到的物理参数（加速训练）——
    privileged_extra_obs_enabled: bool = True    # 启用额外特权观测
    privileged_extra_obs_dim: int = 39           # 额外 39 维
    num_single_privileged_obs = 71               # critic 单帧总观测 71 维
    state_space = 71                             # critic 状态空间维度
    privileged_extra_joint_count: int = 6        # 特权观测含 6 个关节量
    privileged_extra_wheel_count: int = 2        # 2 个轮相关量
    privileged_extra_body_count: int = 1         # 1 个车体量
    privileged_extra_inertia_body_count: int = 0 # 惯量项 0 个
    privileged_extra_material_body_count: int = 2  # 材质(摩擦)项 2 个（左右轮）
    # privileged extra obs 的实体选择使用名字/正则配置，避免依赖 robot 内部 body/joint 顺序。
    privileged_extra_joint_names: tuple[str, ...] = (  # 哪些关节进特权观测
        ".*_rear1_joint",
        ".*_front1_joint",
        ".*_wheel_joint",
    )
    privileged_extra_wheel_body_names: tuple[str, ...] = (  # 哪些轮体进特权观测
        "left_wheel_link",
        "right_wheel_link",
    )
    privileged_extra_body_names: tuple[str, ...] = (        # 哪些连杆进特权观测
        "base_link",
        # *V14_ORDERED_LEG_BODY_NAMES,
        # "left_wheel_link",
        # "right_wheel_link",
        # "gimbal_yaw_link",
        # "gimbal_pitch_link",
    )
    privileged_extra_inertia_body_names: tuple[str, ...] = (  # 惯量观测体（当前空）
        # "base_link",
        # "left_wheel_link",
        # "right_wheel_link",
        # "gimbal_yaw_link",
        # "gimbal_pitch_link",
    )
    privileged_extra_material_body_names: tuple[str, ...] = (  # 摩擦观测体：左右轮
        "left_wheel_link",
        "right_wheel_link",
    )
    obs_input_clip_cfg = V14_EXTRA_OBS_CLIP      # 特权观测的裁剪表（覆盖前面的基础表）
    obs_input_scale_cfg = V14_EXTRA_OBS_SCALE    # 特权观测的缩放表（同上）
    debug_obs_alert_threshold: float = 120.0     # 观测异常报警阈值
    debug_obs_alert_topk: int = 3
    debug_obs_alert_print_interval: int = 1

    # REWARD MAP V14_FLAT:
    # - 仅直接作用于 WheelbipeV14FlatEnvCfg 及继承后未重写 rewards 的 flat 类任务。
    # - Log 只会显示实际进入 reward_terms 且 cfg.rewards 中存在的键。
    # ★★★ 奖励权重表（RL 的灵魂）：策略做什么、不做什么全由它决定 ★★★
    # 正值 = 鼓励该项行为（表现好得正分）；负值 = 惩罚该项行为。
    # 权重越大该项在总奖励里话语权越大；0 = 该项关闭。
    # 各项的计算公式在环境类（V13 父类）里，这里只配"权重×公式"。
    orientation_x_bias = 2.      # 以下一组是姿态奖励的形状参数（σ=高斯宽度，A=幅度…）
    orientation_x_sigma = 3.
    orientation_x_A = 2.
    orientation_y_bias = 2.
    orientation_y_sigma = 3.
    orientation_y_A = 2.
    orientation_x_square_sigma = 4.
    orientation_y_square_sigma = 2.
    flat_pitch_tanh_sigma = 0.1      # 俯仰 tanh 奖励的 σ
    flat_roll_tanh_sigma = 0.05      # 横滚 tanh 奖励的 σ
    foot_bound_dist = 0.12           # 轮子"乱甩"的判定距离 12cm
    foot_bound_square_sigma = 2.
    foot_bound_exp_pen_sigma = 0.2
    foot_bound_exp_sigma = 0.02
    foot_bound_ssquare_sigma = 8.
    lin_vel_xy_sigma = 0.5           # 速度追踪奖励的 σ（误差多大时奖励明显衰减）
    lin_vel_xy_tight_sigma = 0.1     # 严格版 σ
    lin_vel_xy_soft_sigma = 1.5      # 宽松版 σ
    high_speed_pen_sigma = 1.0       # 高速惩罚 σ
    ang_vel_z_sigma = 0.25           # 偏航追踪 σ
    ang_vel_z_square_sigma = 0.5
    high_angVel_pen_sigma = 1.0      # 高角速度惩罚 σ
    height_sigma = 0.025             # 身高追踪 σ=2.5cm
    height_square_sigma = 10.
    base_height_bound = 0.2          # 身高下限 0.2m（低于就罚）
    pen_base_too_low_sigma = 5.
    orientation_y_exp_sigma = 0.02
    orientation_x_exp_sigma = 0.01
    lin_vel_err_constraint = 1.0     # NP3O 约束用的误差限
    ang_vel_err_constraint = 0.8
    height_err_constraint = 0.15
    no_fork_square_sigma = 5.        # 防"劈叉"（两腿岔开）奖励参数
    rewards = OrderedDict(
        termination = -200.,         # ★摔倒终止：一次 -200（最大的罚，让策略极度怕摔）
        leg_joint_acc=-5e-7,         # 腿关节加速度惩罚（动作要柔，别猛甩腿）
        leg_joint_vel = -5.0e-3,     # 腿关节速度惩罚
        leg_joint_pair_pos_diff=-0.0, # 左右腿对称性惩罚（当前关闭）
        joint_torque=-1e-4,          # 力矩惩罚（省电+保护电机）
        wheel_acc=-1e-8,             # 轮加速度惩罚（轮子转得平顺）
        wheel_vel=-1e-5,             # 轮速惩罚
        wheel_power=-1e-4,           # 轮功率惩罚（直接对应电池功耗）
        wheel_air_spin=0.,           # 腾空时轮子空转惩罚（当前关闭）
        lin_vel_z=-0.5,              # 竖直速度惩罚（别上下颠簸/蹦跳）
        ang_vel_xy=-0.05,            # 横滚/俯仰角速度惩罚（车身要稳）
        action_smoothness_leg=-0.05, # 腿动作平滑性惩罚（相邻动作差值）
        action_rate = -0.01,         # 动作变化率惩罚
        action_smoothness_wheel=-0.01, # 轮动作平滑性惩罚
        flat_orientation_y=-0.0,     # 俯仰保持水平奖励（当前关闭）
        flat_orientation_y_v=-2.0,   # 俯仰角速度惩罚
        flat_orientation_y_exp = 1.0,  # 俯仰 exp 奖励（越平越好）
        # flat_pitch_l1 = -1.0,
        # flat_pitch_tanh = 1.0,
        flat_orientation_x=-0.0,     # 横滚保持水平奖励（当前关闭）
        flat_orientation_x_v=-2.0,   # 横滚角速度惩罚
        flat_orientation_x_exp = 1.0,  # 横滚 exp 奖励
        # flat_roll_l1 = -1.0,
        # flat_roll_tanh = 1.0,
        track_lin_vel_xy=1.0,        # ★追踪平移速度指令（主线任务：让走哪就走哪）
        track_lin_vel_xy_tight=0.0,  # 严格版速度追踪（当前关闭）
        track_lin_vel_xy_square=-1.0,  # 速度误差平方惩罚（跟得不准就罚）
        # track_lin_vel_xy_square=-0.1,
        track_ang_vel_z=1.0,         # ★追踪偏航角速度指令（转向控制）
        track_ang_vel_z_square=-1.0, # 转向误差平方惩罚
        # track_ang_vel_z_square=-0.1,
        stand_still_lin_vel=-1.0,    # 指令为零时乱动惩罚（站着别晃）
        # stand_still=-2.0,
        stand_still=-0.0,            # 站住奖励（当前关闭）
        track_height_exp=0.0,        # 身高追踪 exp 奖励（基础版关闭，课程任务里开）
        track_height_exp_soft=0.0,   # 宽松版身高追踪（关闭）
        track_height_exp_tight=1.0,  # ★严格版身高追踪（打开：身高要准）
        track_height_square=-1.0,    # 身高误差平方惩罚
        track_height_exp_both_wheels_contact=0.0,  # 双轮着地时的身高追踪（关闭）
        no_fork = -1.0,              # 防劈叉惩罚
        no_fork_square = -1.0,       # 防劈叉平方惩罚
        no_fork_exp=-0.0,            # 防劈叉 exp（关闭）
        no_fork_z_exp=-0.0,          # 防劈叉 z 向 exp（关闭）
        undesired_contact=-2.0,      # ★不该碰的部件碰地惩罚（车体/腿蹭地）
    )

    def __post_init__(self):
        # __post_init__：配置对象构造完成后的"最后一道加工"——
        # 根据开关的组合关系修正其它参数（如关掉不兼容的传感器、扩观测维度、搭命令生成器）。
        super().__post_init__()      # 先执行父类(25v3)的后处理
        _apply_v14_flat_runtime_optimizations(self)   # 应用 V14 平地任务的性能优化（cfg_utils 里定义）
        # if getattr(self.terrain, "terrain_type", None) == "plane":
        #     self.terrain = copy.deepcopy(self.terrain)
        #     self.terrain.physics_material = None
        if bool(getattr(self, "use_leg_length_as_height", False)):
            # 若用"腿长"当高度：改用相对高度口径，并关闭状态机/轮前扫描（不兼容）
            self.use_absolute_height = False
            self.enable_state_machines = False
            self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)  # 深拷贝避免改动共享配置
            self.airborne_state_machine_cfg["enabled"] = False
            self.wheel_forward_scan_cfg = copy.deepcopy(self.wheel_forward_scan_cfg)
            self.wheel_forward_scan_cfg["enabled"] = False

        self._apply_ctrl_mode_obs_cfg()   # 按配置把 7 维"控制模式观测"计入观测/状态维度

        if bool(getattr(self, "use_absolute_height", False)):
            # 绝对高度口径：不需要测地面，关掉车身和轮子上的高度扫描仪（省大量算力）
            self.height_scanner = None
            _disable_v14_wheel_height_scanners(self)
        else:
            _enable_v14_body_height_scanner(self)   # 相对高度口径：必须开车身扫描仪
            if bool(self.airborne_state_machine_cfg.get("enabled", False)) or bool(
                getattr(self, "wheel_forward_scan_cfg", {}).get("enabled", False)
            ) or bool(
                getattr(self, "stair_state_machine_cfg", {}).get("enabled", False)
            ) or bool(
                getattr(self, "jump_takeoff_state_machine_cfg", {}).get("enabled", False)
            ):
                _enable_v14_wheel_height_scanners(self)   # 有状态机/扫描需求时轮子扫描仪也要开
            else:
                _disable_v14_wheel_height_scanners(self)

        if hasattr(self, "use_frame_stack") and self.use_frame_stack:
            # 开帧堆叠时：策略观测维度 = 历史长度 × 单帧维度(28)
            self.observation_space = self.num_obs_hist * getattr(self, "num_single_obs", 28)
        self.state_space = self.num_privileged_obs_hist * getattr(self, "num_single_privileged_obs", 32)  # critic 状态维度

        # self.commands = mdp.UniformVelocityCommandCfg(   # （注释掉：普通速度命令生成器，已被下面的特殊模式版替代）
        #     asset_name="robot",
        #     resampling_time_range=(7.0, 15.0),
        #     rel_standing_envs=0.1,
        #     rel_heading_envs=0.5,
        #     heading_command=True,
        #     heading_control_stiffness=1.0,
        #     debug_vis=False,
        #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #         lin_vel_x=(-2.5, 2.5),
        #         lin_vel_y=(0.0, 0.0),
        #         ang_vel_z=(-torch.pi, torch.pi),
        #         heading=(-torch.pi, torch.pi),
        #     ),
        # )

        # ── SpecialModeUniformVelocityCommand ──
        # 各特殊模式按 rel_envs 占非站立环境的比例独立分配（互斥、无优先级）。
        # 每个 env 单独掷 U(0,1) 选桶，兼容逐 env 重采样；模式顺序每次随机打乱。
        # 模式可配置 iteration_start/iteration_end 按训练轮次启停；
        # iteration 由 env 根据 common_step_counter 外推，无需 runner 每轮回调。
        # 模式 ranges 的每个字段支持单区间 ``(low, high)`` 或多区间 ``[(l1,h1), (l2,h2)]``，
        # 多区间时按宽度比例随机选一段再均匀采样。
        # ★速度指令生成器：每隔几秒给每个环境发一条"运动指令"，并支持特殊训练模式
        self.commands = mdp.SpecialModeUniformVelocityCommandCfg(
            asset_name="robot",               # 作用对象：机器人
            resampling_time_range=(5.0, 15.0),  # 每 5~15 秒随机重采一次指令
            rel_standing_envs=0.1,            # 10% 概率指令为"站住别动"
            rel_heading_envs=0.5,             # 50% 概率发"朝向目标"而非直接发角速度
            heading_command=True,             # 启用朝向命令模式
            heading_control_stiffness=5.0,    # 朝向跟踪的 P 增益
            debug_vis=False,                  # 不画指令箭头
            special_mode_min_episode_time=5.0,   # 进特殊模式前本局至少已进行 5 秒
            special_mode_require_stable=False,   # 进入特殊模式是否要求机身先稳定
            special_mode_stable_projected_gravity_xy_norm_max=0.5,   # 稳定判据：重力投影 xy 范数
            special_mode_stable_root_lin_vel_b_abs_max=3.0,          # 稳定判据：线速度上限
            special_mode_stable_root_ang_vel_b_abs_max=10.0,         # 稳定判据：角速度上限
            ranges=mdp.SpecialModeUniformVelocityCommandCfg.Ranges(  # 普通指令的采样范围
                lin_vel_x=(-2.7, 2.7),        # x 速度 ±2.7 m/s
                lin_vel_y=(0.0, 0.0),         # y 速度 0（轮腿机器人横移靠平移模式）
                ang_vel_z=(-2.*torch.pi, 2.*torch.pi),  # 偏航角速度 ±2π
                heading=(-torch.pi, torch.pi),  # 目标朝向范围
            ),
            special_modes={                   # 特殊训练模式表（按比例分配给环境）
                # 模式0 — 纯自旋：20% 非站立环境
                "spin_low": mdp.SpecialModeEntryCfg(   # 低速自旋：原地打转（小陀螺入门）
                    rel_envs=0.15,              # 15% 的环境练这个
                    iteration_start=3000,       # 第 3000 轮后才启用（先学会走再学转）
                    iteration_end=-1,      # 永不过期
                    disable_jump_takeoff=False,
                    debug_print=False,
                    ranges=mdp.SpecialModeEntryCfg.Ranges(
                        lin_vel_x=(-0.1, 0.1),  # 几乎不平移
                        lin_vel_y=(0.0, 0.0),
                        ang_vel_z=[(2.*torch.pi, 3.25*torch.pi), (-3.25*torch.pi, -2.*torch.pi)],  # 自旋 2~3.25π（正反两方向）
                    ),
                ),
                "spin_mid": mdp.SpecialModeEntryCfg(   # 中速自旋（更快）
                    rel_envs=0.15,
                    iteration_start=4000,
                    iteration_end=-1,      # 永不过期
                    disable_jump_takeoff=False,
                    debug_print=False,
                    ranges=mdp.SpecialModeEntryCfg.Ranges(
                        lin_vel_x=(-0.1, 0.1),
                        lin_vel_y=(0.0, 0.0),
                        ang_vel_z=[(3.25*torch.pi, 4.5*torch.pi), (-4.5*torch.pi, -3.25*torch.pi)],  # 自旋 3.25~4.5π
                    ),
                ),
                # "spin_high": mdp.SpecialModeEntryCfg(  # （注释掉：高速自旋模式）
                #     rel_envs=0.1,
                #     iteration_start=5000,
                #     iteration_end=-1,      # 永不过期
                #     debug_print=False,
                #     ranges=mdp.SpecialModeEntryCfg.Ranges(
                #         lin_vel_x=(-0.1, 0.1),
                #         lin_vel_y=(0.0, 0.0),
                #         ang_vel_z=[(4.5*torch.pi, 5.5*torch.pi), (-4.5*torch.pi, -5.5*torch.pi)],
                #     ),
                # ),
                # 模式1 — 高速前冲/后退：20% 非站立环境
                "dash": mdp.SpecialModeEntryCfg(       # 冲刺：±2~3 m/s 的高速机动
                    rel_envs=0.3,
                    iteration_start=2000,
                    iteration_end=-1,
                    disable_jump_takeoff=True,   # 冲刺时禁止触发跳跃
                    debug_print=False,
                    ranges=mdp.SpecialModeEntryCfg.Ranges(
                        lin_vel_x=[(2.0, 3.0), (-3.0, -2.0)],  # 多区间：向前或向后高速
                        lin_vel_y=(0.0, 0.0),
                        ang_vel_z=[(-2.*torch.pi, 2.*torch.pi)],
                    ),
                ),
            },
        )
        self.height_command_special_modes_cfg = {   # 身高指令的特殊模式（正弦/阶跃变高训练），当前关闭
            "enabled": False,
            "min_episode_time": 0.0,
            "modes": {
                "height_sine": {               # 正弦波变高：身高按正弦规律变化
                    "rel_envs": 0.0,           # 分配比例（当前 0 = 关）
                    "iteration_start": 0,
                    "iteration_end": -1,
                    "height_wave": mdp.HeightWaveCfg(   # 正弦波参数
                        mean=0.3,                  # 平均身高 0.3m
                        mean_range=(0.25,0.35),    # 均值随机范围
                        amplitude=0.1,             # 振幅 10cm
                        amplitude_range=(0.05,0.1),
                        frequency_hz=1.0,          # 频率 1Hz
                        frequency_range_hz=(1.0,2.5),
                        phase=0.0,                 # 相位
                        random_phase=True,         # 相位随机
                        clamp_range=(0.20, 0.40),  # 身高夹在 0.2~0.4m
                    ),
                },
                "height_step": {               # 阶跃变高：身高突然跳变（类似方波）
                    "rel_envs": 0.0,
                    "iteration_start": 0,
                    "iteration_end": -1,
                    "height_step": mdp.HeightStepCfg(   # 阶跃波参数（字段同上）
                        mean=0.3,
                        mean_range=(0.25, 0.35),
                        amplitude=0.05,
                        amplitude_range=(0.1, 0.2),
                        frequency_hz=1.0,
                        frequency_range_hz=(1.0, 2.5),
                        phase=0.0,
                        random_phase=True,
                        clamp_range=(0.20, 0.40),
                    ),
                },
            },
        }

        self.decimation = 4              # 每个策略步内跑 4 次物理仿真
        self.sim.dt = 1 / 200.0          # 物理仿真步长 1/200 秒 → 策略频率 = 200/4 = 50Hz
        self.max_wheel_torque = 20.0     # 轮电机最大力矩 20 N·m

    def _apply_ctrl_mode_obs_cfg(self, enabled: bool | None = None):
        # 把"控制模式观测"(7维)并入观测维度：维护 num_single_obs / 空间维度的一致性
        # ctrl_mode_obs 独立于状态机 cfg 生效（jump_takeoff_state_machine_cfg 已移除）。
        extra_obs_cfg: dict = {}
        use_extra_obs = (                # 是否启用：显式参数 > ctrl_mode_obs_enabled > 旧名开关
            bool(
                getattr(
                    self,
                    "ctrl_mode_obs_enabled",
                    getattr(self, "jump_takeoff_extra_obs_enabled", False),
                )
            )
            if enabled is None
            else bool(enabled)
        )
        extra_obs_dim = int(             # 维度（默认 7）
            getattr(self, "ctrl_mode_obs_dim", getattr(self, "jump_takeoff_extra_obs_dim", 7))
        )
        extra_obs_layout = tuple(        # 布局（各维含义）
            getattr(
                self,
                "ctrl_mode_obs_layout",
                getattr(
                    self,
                    "jump_takeoff_extra_obs_layout",
                    (
                        "normal",
                        "stair",
                        "slope",
                        "recover",
                        "jump",
                        "height_target",
                        "state_time",
                    ),
                ),
            )
        )
        extra_obs_cfg.update(            # 汇总成字典
            {
                "enabled": use_extra_obs,
                "dim": extra_obs_dim,
                "layout": extra_obs_layout,
            }
        )
        self.ctrl_mode_obs_enabled = use_extra_obs    # 把最终决定写回配置（统一来源）
        self.ctrl_mode_obs_dim = extra_obs_dim
        self.ctrl_mode_obs_layout = extra_obs_layout
        # Backward-compatible mirrors for older state-machine/config code.
        # 同步镜像到旧字段名（兼容旧代码）
        self.jump_takeoff_extra_obs_enabled = use_extra_obs
        self.jump_takeoff_extra_obs_dim = extra_obs_dim
        self.jump_takeoff_extra_obs_layout = extra_obs_layout
        if not use_extra_obs:
            return                     # 未启用：维度不用动

        current_extra_dim = int(getattr(self, "_ctrl_mode_obs_applied_dim", 0))  # 已经并入过多少维（防重复叠加）
        extra_obs_delta = max(extra_obs_dim - current_extra_dim, 0)  # 本次需要新增的维数
        self._ctrl_mode_obs_applied_dim = max(current_extra_dim, extra_obs_dim)
        self._jump_takeoff_extra_obs_applied_dim = self._ctrl_mode_obs_applied_dim
        if extra_obs_delta > 0:
            self.num_single_obs = int(getattr(self, "num_single_obs", 28)) + extra_obs_delta  # 单帧观测维度 +7
            self.num_single_privileged_obs = (
                int(getattr(self, "num_single_privileged_obs", 32)) + extra_obs_delta  # critic 单帧也 +7
            )
            if isinstance(getattr(self, "observation_space", None), dict):
                observation_space = dict(self.observation_space)   # 观测空间是字典形式时逐组加
                if "policy" in observation_space:
                    observation_space["policy"] = int(observation_space["policy"]) + extra_obs_delta
                if "critic" in observation_space:
                    observation_space["critic"] = int(observation_space["critic"]) + extra_obs_delta
                self.observation_space = observation_space
            else:
                self.observation_space = int(                      # 整数形式直接加
                    getattr(self, "observation_space", self.num_single_obs)
                ) + extra_obs_delta
            if isinstance(getattr(self, "state_space", None), dict):
                state_space = dict(self.state_space)               # 状态空间同理
                if "critic" in state_space:
                    state_space["critic"] = int(state_space["critic"]) + extra_obs_delta
                self.state_space = state_space
            elif getattr(self, "state_space", None):
                self.state_space = int(self.state_space) + extra_obs_delta
            else:
                self.state_space = (                               # 没定义过就按"历史×单帧"算
                    int(getattr(self, "num_privileged_obs_hist", 1))
                    * self.num_single_privileged_obs
                )
        scale_cfg = dict(getattr(self, "obs_input_scale_cfg", {}))  # 模式观测的缩放系数
        scale_cfg["ctrl_mode_obs"] = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0]  # 第 6 维(身高目标)×5（数值太小放大）
        scale_cfg.pop("jump_takeoff_obs", None)
        self.obs_input_scale_cfg = scale_cfg

    def _apply_jump_takeoff_extra_obs_cfg(self, enabled: bool | None = None):
        """Backward-compatible alias; ctrl_mode_obs is the canonical name."""
        # 旧接口别名：转调新的 _apply_ctrl_mode_obs_cfg
        self._apply_ctrl_mode_obs_cfg(enabled=enabled)


# ─────────────────────────────────────────────────────────────────────────────
# _v2：平地 + 云台航向锁定（PD 控制头部朝向）+ 小陀螺平移（边自旋边按云台指向平移）
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class WheelbipeV14FlatEnvCfg_v2(WheelbipeV14FlatEnvCfg):
    """Flat V14 with gimbal-heading PD and gimbal-frame spin/translation commands."""

    # —— 云台航向锁定 PD 控制参数 ——
    gimbal_heading_control_cfg = {
        "enabled": True,                  # 启用航向锁定
        "target_mode": "sampled",         # 目标朝向来源：每局随机采样
        "fixed_heading": 0.0,             # fixed 模式用的固定角
        "heading_range": (-torch.pi, torch.pi),  # 随机朝向的采样范围
        "kp": 20.0,                       # PD 比例增益（默认值）
        "kd": 0.1,                        # PD 微分增益
        "kp_range": (20.0, 40.0),         # kp 随机化范围（域随机化）
        "kd_range": (0.05, 0.1),          # kd 随机化范围
        "randomize_gains": True,          # 开启增益随机化
        "gain_distribution": "uniform",   # 均匀分布
        "max_effort": 2.0,                # PD 输出力矩限幅 2 N·m
        "apply_only_in_special_mode": False,  # False=全程锁定；True=仅特殊模式时锁定
        "special_mode_name": "gimbal_spin_translate",  # 限定模式名
    }
    # —— 小陀螺平移模式参数 ——
    gimbal_spin_translate_cfg = {
        "enabled": True,                  # 启用该模式
        "special_mode_name": "gimbal_spin_translate",  # 对应命令生成器里的模式名
        # "lin_vel_yaw_speed_range": [(0.0, 0.04), (0.25, 0.5)],
        "lin_vel_yaw_speed_range": [(0.0, 0.75)],   # 平移速度采样：0~0.75 m/s（多段写法备用见上）
        "lin_vel_yaw_speed_deadzone": 0.05,         # 死区 0.05：低于此速度视为"不移动"
        "lin_vel_yaw_heading_range": (-torch.pi, torch.pi),  # 平移方向随机全向
        "lin_vel_yaw_height_range": (0.20, 0.40),   # 模式内身高指令随机 0.2~0.4m
        "zero_heading_in_deadzone": False,          # 死区内方向观测是否置零
        "use_sampled_heading_obs": False,           # 观测方向用采样的还是现算的
        "require_heading_control": True,            # 需要航向锁定配合
        "project_to_body_command": False,           # False=不做车体系投影（策略自己处理）
    }
    # 该模式下要屏蔽的"普通模式"奖励项（避免两套目标打架）
    gimbal_spin_suppressed_reward_terms = (
        "track_lin_vel_xy",           # 普通速度追踪
        "track_lin_vel_xy_soft",
        "track_lin_vel_xy_tight",
        "track_lin_vel_xy_huge_gap",
        "track_lin_vel_xy_square",
        "stand_still_lin_vel",        # 站住惩罚（自旋时车体本来在动）
    )
    # 模式观测的 7 维布局改为小陀螺语义
    ctrl_mode_obs_layout = (
        "normal_mode_flag",                     # 0: 普通模式标志
        "gimbal_spin_translate_mode_flag",      # 1: 小陀螺平移模式标志
        "gimbal_spin_speed_cmd_yaw",            # 2: 指令速度大小
        "gimbal_spin_sin_heading_cmd_yaw",      # 3: 指令方向 sin
        "gimbal_spin_cos_heading_cmd_yaw",      # 4: 指令方向 cos
        "gimbal_spin_sin_yaw_joint_angle",      # 5: 云台自转角 sin
        "gimbal_spin_cos_yaw_joint_angle",      # 6: 云台自转角 cos
    )
    # gimbal_spin_track_lin_vel_yaw_frame: reward exp(-||v_cmd_yaw - v_meas_yaw||^2 / sigma).
    gimbal_spin_lin_vel_yaw_sigma = 0.25        # 速度向量追踪奖励的 σ
    # gimbal_spin_track_lin_speed: reward exp(-(speed_cmd - speed_meas)^2 / sigma).
    gimbal_spin_lin_speed_sigma = 0.25          # 速率追踪奖励的 σ
    # gimbal_spin_track_lin_heading: reward exp(-heading_error^2 / sigma), gated by valid speed.
    gimbal_spin_lin_heading_sigma = 0.025       # 方向追踪奖励的 σ（方向要准，σ 小）
    # Minimum command speed required before applying heading-direction reward/penalty.
    gimbal_spin_heading_cmd_speed_min = 0.1     # 指令速度门限：低于此不计方向分
    # Minimum measured speed required before applying heading-direction reward/penalty.
    gimbal_spin_heading_meas_speed_min = 0.0    # 实测速度门限
    # gimbal_spin_lin_vel_yaw_square: penalty sigma^2 * ||v_cmd_yaw - v_meas_yaw||^2.
    gimbal_spin_lin_vel_yaw_square_sigma = 0.5  # 速度误差平方惩罚的 σ
    # gimbal_spin_lin_speed_overshoot: penalty max(speed_meas - speed_cmd, 0)^2 * sigma^2.
    gimbal_spin_lin_speed_overshoot_sigma = 0.5 # 超速惩罚的 σ
    # gimbal_spin_heading_error_square: penalty sigma^2 * heading_error^2, gated by valid speed.
    gimbal_spin_heading_error_square_sigma = 4. # 方向误差平方惩罚的 σ
    # gimbal_spin_stand_still_lin_vel: L1 yaw-link-frame linear velocity penalty when speed_cmd is near zero.
    gimbal_spin_stand_still_speed_threshold = 0.05  # "要求站住"的速度阈值

    def __post_init__(self):
        super().__post_init__()       # 先跑基类的后处理
        self.robot_cfg = copy.deepcopy(self.robot_cfg)   # 深拷贝机器人配置（要改它的执行器参数）
        gimbal_yaw_actuator = self.robot_cfg.actuators.get("gimbal_yaw", None)  # 找到 yaw 执行器
        if gimbal_yaw_actuator is not None:
            gimbal_yaw_actuator.effort_limit = float(self.gimbal_heading_control_cfg.get("max_effort", 5.0))  # 执行器力矩上限对齐 PD 限幅
        # 自旋/冲刺模式提前开练（去掉基类设的 iteration 门槛）
        self.commands.special_modes['spin_low'].iteration_start = 0   # 低速自旋从第 0 轮就练
        self.commands.special_modes['spin_low'].rel_envs = 0.1        # 占 10%
        self.commands.special_modes['spin_mid'].iteration_start = 0   # 中速自旋同上
        self.commands.special_modes['spin_mid'].rel_envs = 0.1
        self.commands.special_modes['dash'].iteration_start = 0       # 冲刺同上
        self.commands.special_modes['dash'].rel_envs = 0.2            # 占 20%
        # 给命令生成器添加 PD 增益随机化事件 + 新特殊模式
        heading_cfg = dict(self.gimbal_heading_control_cfg)
        kp_range = heading_cfg.get("kp_range", None) if bool(heading_cfg.get("randomize_gains", False)) else None  # 开随机才取范围
        kd_range = heading_cfg.get("kd_range", None) if bool(heading_cfg.get("randomize_gains", False)) else None
        self.events = copy.deepcopy(self.events)      # 深拷贝事件表（要往里加项）
        self.events.gimbal_heading_pd_gains = EventTerm(  # 新事件：开局随机化每个环境的 PD 增益
            func=mdp.randomize_gimbal_heading_pd_gains,
            mode="startup",
            params={
                "kp_distribution_params": kp_range,
                "kd_distribution_params": kd_range,
                "distribution": heading_cfg.get("gain_distribution", "uniform"),
            },
        )
        self.commands = copy.deepcopy(self.commands)  # 深拷贝命令配置（要往里加模式）
        special_modes = getattr(self.commands, "special_modes", {}) or {}
        if not isinstance(special_modes, dict):       # 兼容旧格式（列表→字典）
            special_modes = {
                f"mode_{idx}": mode_cfg
                for idx, mode_cfg in enumerate(tuple(special_modes))
            }
        else:
            special_modes = dict(special_modes)
        special_modes["gimbal_spin_translate"] = mdp.SpecialModeEntryCfg(  # ★新特殊模式：小陀螺平移
            rel_envs=0.2,                 # 20% 环境练这个
            iteration_start=0,
            iteration_end=-1,
            disable_jump_takeoff=True,    # 该模式下禁跳跃
            debug_print=False,
            ranges=mdp.SpecialModeEntryCfg.Ranges(
                lin_vel_x=(0.0, 0.0),     # 该模式的普通指令置零（速度由 spin_translate 机制接管）
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=[               # 同时给一个自旋角速度指令
                    (2.4 * torch.pi, 3.6 * torch.pi),
                    (-3.6 * torch.pi, -2.4 * torch.pi),
                ],
            ),
        )
        self.commands.special_modes = special_modes
        self.ctrl_mode_obs_layout = tuple(self.ctrl_mode_obs_layout)   # 布局转成元组
        self.jump_takeoff_extra_obs_layout = self.ctrl_mode_obs_layout  # 同步旧字段
        scale_cfg = dict(getattr(self, "obs_input_scale_cfg", {}))
        scale_cfg["ctrl_mode_obs"] = [1.0] * 7     # 该布局下 7 维都不缩放
        self.obs_input_scale_cfg = scale_cfg
        self.stand_still_deadzone_enabled = True
        # —— 下面给"小陀螺专属奖励项"配权重（公式在 env.py 的 _get_gimbal_spin_translate_reward_terms）——
        self.rewards["stand_still_lin_vel"] = -1.0
        # self.rewards["track_lin_vel_xy_square"] = -0.1
        # self.rewards["track_ang_vel_z_square"] = -0.1
        self.rewards["gimbal_spin_track_lin_vel_yaw_frame"] = 1.    # 追踪云台系速度向量 +1
        self.rewards["gimbal_spin_track_lin_speed"] = 1.            # 追踪速率 +1
        self.rewards["gimbal_spin_track_lin_heading"] = 5.0         # 追踪方向 +5（方向最重要）
        self.rewards["gimbal_spin_lin_vel_yaw_square"] = -0.2       # 速度误差平方 -0.2
        self.rewards["gimbal_spin_lin_speed_overshoot"] = -0.       # 超速惩罚（当前关闭）
        self.rewards["gimbal_spin_heading_error_square"] = -0.2     # 方向误差平方 -0.2
        # self.rewards["gimbal_spin_track_lin_heading_v2"] = 5.0
        # self.rewards["gimbal_spin_heading_error_square_v2"] = -0.2
        self.rewards["gimbal_spin_stand_still_lin_vel"] = -1.0      # 要求站住时乱动 -1


@configclass
class WheelbipeV14FlatEnvCfg_v2_Play(WheelbipeV14FlatEnvCfg_v2):
    """Play config for V14 flat v2 gimbal spin/translate evaluation."""

    # _v2 的演示配置：固定参数、所有环境都进小陀螺平移模式，方便观察效果
    events = EventCfgV14_Play()       # 换 Play 事件表
    curriculum = None                 # 演示不需要课程
    use_frame_stack = False
    num_obs_hist = 1
    num_privileged_obs_hist = 1

    def __post_init__(self):
        super().__post_init__()
        self.play = True              # 标记 Play 模式（环境据此关闭随机化/课程等）
        self.episode_length_s = 20.0  # 每局 20 秒
        self.height_range = [0.2, 0.3]   # 身高范围收窄
        self.play_gimbal_spin_translate_debug_vis = True   # 打开头顶提示球
        self.play_gimbal_spin_translate_marker_height = 0.85  # 球的高度 0.85m
        self.play_gimbal_spin_translate_marker_radius = 0.12  # 球半径 0.12m

        # 航向锁定改为固定角（演示时不随机）
        self.gimbal_heading_control_cfg = dict(self.gimbal_heading_control_cfg)
        self.gimbal_heading_control_cfg['target_mode'] = 'fixed'
        # 小陀螺平移改为固定参数：恒速 0.6、朝正前方
        self.gimbal_spin_translate_cfg = dict(self.gimbal_spin_translate_cfg)
        self.gimbal_spin_translate_cfg["enabled"] = True
        self.gimbal_spin_translate_cfg["lin_vel_yaw_speed_range"] = (0.6, 0.6)   # 恒速 0.6 m/s
        self.gimbal_spin_translate_cfg["lin_vel_yaw_speed_deadzone"] = 0.05
        self.gimbal_spin_translate_cfg["lin_vel_yaw_heading_range"] = (0., 0.)   # 方向固定 0（正前）
        self.gimbal_spin_translate_cfg["project_to_body_command"] = False

        # 命令生成器：其它特殊模式全部关掉，只留小陀螺平移且占 100%
        self.commands = copy.deepcopy(self.commands)
        special_modes = getattr(self.commands, "special_modes", {}) or {}
        if not isinstance(special_modes, dict):
            special_modes = {
                f"mode_{idx}": mode_cfg
                for idx, mode_cfg in enumerate(tuple(special_modes))
            }
        else:
            special_modes = dict(special_modes)
        for mode_cfg in special_modes.values():
            mode_cfg.rel_envs = 0.0           # 其它模式比例清零
            mode_cfg.iteration_start = 0
            mode_cfg.iteration_end = -1
        if "gimbal_spin_translate" in special_modes:
            special_modes["gimbal_spin_translate"].rel_envs = 1.0   # 小陀螺平移占 100%
        self.commands.special_modes = special_modes


def _apply_v14_airborne_landing_precontact_cfg(cfg) -> None:
    """Apply the airborne landing/pre-contact additions shared by flat-v1 and rough-v1."""

    # 共享工具函数：给"腾空预训练"类任务配置落地/预接触阶段的附加奖励项。
    # flat_v1 和 rough_v1 都调用它，避免两处重复写同样的字典。
    cfg.airborne_state_machine_cfg = copy.deepcopy(cfg.airborne_state_machine_cfg)  # 深拷贝（要改内容）

    landing_trajectory = copy.deepcopy(              # 覆盖落地缓冲轨迹参数
        cfg.airborne_state_machine_cfg.get("landing_trajectory", {})
    )
    landing_trajectory.update(
        {
            "enabled": False,                        # 轨迹引导当前关闭（用奖励隐式引导）
            "start_wheel_contact_duration_s": 0.02,
            "target_height": 0.24,                   # 轨迹终点身高 0.24m
            "end_vel_z": 0.0,
            "min_height_margin": 0.02,
            "min_down_vel": 0.2,
            "duration_s": 0.3,
            "max_abs_acc": 30.0,
        }
    )
    cfg.airborne_state_machine_cfg["landing_trajectory"] = landing_trajectory

    reward_additions = dict(cfg.airborne_state_machine_cfg.get("reward_additions", {}))  # 附加奖励项定义表
    # 每项："名字" → {"type": 环境里对应的奖励计算类型, 其它参数}
    reward_additions["airborne_wheel_contact_force_over"] = {   # 落地冲击力过大惩罚
        "type": "wheel_contact_force_over",
        "force_threshold": 300.0,             # 冲击超过 300N 开始罚
        "mode": "l1",                         # 线性(L1)惩罚
        "reduce": "sum",                      # 双轮求和
    }
    reward_additions["airborne_landing_wheel_body_x_positive"] = {  # 落地时轮子相对车体要"往前伸"
        "type": "landing_wheel_body_x_positive",
        "target_x": 0.03,                     # 目标前伸 3cm
        "sigma": 0.03,
        "command_x_min": 1.0,                 # 仅当指令速度 ≥1 时启用（往前跳才前伸）
        "start_wheel_contact_duration_s": 0.02,
        "contact_force_threshold": 1.0,
        "contact_mode": "any_wheel",          # 任一轮触发即可
        "use_entry_command": True,            # 用进入腾空时刻的指令判断
    }
    reward_additions["airborne_air_wheel_zero_torque_exp"] = {  # 空中轮子别乱给力矩（收着）
        "type": "wheel_zero_torque_exp",
        "sigma": 1.5,
        "before_wheel_contact_duration_s": 0.02,
        "contact_mode": "any_wheel",
    }
    reward_additions["airborne_precontact_wheel_directional_speed"] = {  # 预接触阶段轮子要预先加速到指令方向
        "type": "wheel_directional_speed",
        "start": 0.0,
        "full": 10.0,                         # 满额奖励尺度
        "command_x_threshold": 1.0,           # 指令速度 ≥1 才启用
        "root_x_threshold": 1.0,
        "reduce": "min",                      # 取双轮较差者
        "before_wheel_contact_duration_s": 0.02,
        "contact_mode": "any_wheel",
    }
    reward_additions["airborne_precontact_wheel_directional_speed_shortfall"] = {  # 上面没做够的罚分版
        "type": "wheel_directional_speed_shortfall",
        "start": 0.0,
        "full": 10.0,
        "command_x_threshold": 1.0,
        "root_x_threshold": 1.0,
        "reduce": "min",
        "require_wheel_contact_timer_started": True,
        "before_wheel_contact_duration_s": 0.05,
        "contact_mode": "any_wheel",
    }
    reward_additions["airborne_landing_wheel_max_contact_force"] = {  # 落地最大冲击力惩罚（软着陆）
        "type": "landing_wheel_max_contact_force",
        "force_start": 200.0,                 # 200N 开始罚
        "force_full": 400.0,                  # 400N 罚满
        "start_wheel_contact_duration_s": 0.02,
        "contact_force_threshold": 1.0,
        "contact_mode": "any_wheel",
    }
    reward_additions["airborne_landing_traj_height"] = {  # （若开轨迹）追踪轨迹高度
        "type": "landing_traj_height_exp",
        "sigma": 0.01,
    }
    reward_additions["airborne_landing_traj_vel_z"] = {   # （若开轨迹）追踪轨迹竖直速度
        "type": "landing_traj_vel_z_exp",
        "sigma": 0.25,
    }
    cfg.airborne_state_machine_cfg["reward_additions"] = reward_additions

    # 保持 Rough-v1 当前启用的 airborne 权重；其它候选项仅保留 reward_additions，按需再开。
    # —— 上面定义的附加项，只有在这里配了权重才真正生效 ——
    # cfg.rewards["airborne_wheel_contact_force_over"] = -0.1
    # cfg.rewards["airborne_landing_wheel_body_x_positive"] = 5.0
    cfg.rewards["airborne_air_wheel_zero_torque_exp"] = 20.0          # 空中轮子零力矩奖励 +20
    cfg.rewards["airborne_precontact_wheel_directional_speed"] = 10.0  # 预加速奖励 +10
    cfg.rewards["airborne_precontact_wheel_directional_speed_shortfall"] = -10.0  # 预加速不足 -10
    # cfg.rewards["airborne_landing_wheel_max_contact_force"] = -10.0
    # cfg.rewards["airborne_landing_traj_height"] = 20.0
    # cfg.rewards["airborne_landing_traj_vel_z"] = 2.0

''' 腾空落地预训练 '''
@configclass
class WheelbipeV14FlatEnvCfg_v1(WheelbipeV14FlatEnvCfg):
    # ★腾空落地预训练任务：把机器人从空中随机扔下来，学会"安全落地"
    termination_duration_steps = 10   # 摔倒判定放宽到连续 10 步（落地瞬间姿态本来就不稳）
    ctrl_mode_obs_enabled = True
    ctrl_mode_obs_dim = 7
    ctrl_mode_obs_layout = (          # 模式观测用普通布局
        "normal",
        "stair",
        "slope",
        "recover",
        "jump",
        "height_target",
        "state_time",
    )
    jump_takeoff_extra_obs_enabled = False   # 旧字段关闭（新字段已开）
    jump_takeoff_extra_obs_dim = 7
    jump_takeoff_extra_obs_layout = ctrl_mode_obs_layout
    height_obs_clip_enabled = True    # 高度观测裁剪打开（腾空高度可能很大）
    height_obs_clip_range = [0.05, 0.45]   # 夹在 5~45cm
    rear2_rear1_joint_limit_lower = -1.0 / 180.0 * torch.pi   # 关节限位奖励参数（v1 微调版）
    rear2_rear1_joint_limit_upper = 68.5 / 180.0 * torch.pi
    rear2_rear1_joint_limit_boundary_ratio = 0.03
    rear2_rear1_joint_limit_lower_boundary_ratio = 0.1   # 下限边界比例放宽
    rear2_rear1_joint_limit_upper_boundary_ratio = 0.05
    rear2_rear1_joint_limit_vel_threshold = 3*torch.pi
    predefined_reset_air = {          # ★空中重置姿势表：重置时把机器人扔在半空
        "enabled": True,
        "modes": (                    # 模式列表（按 prob 概率抽取）
            {
                "name": "air_1",      # 模式：随机高度/姿态/速度扔下来
                "prob": 0.3,          # 30% 概率
                "iteration_start": 0, # 从第 0 轮就启用
                "iteration_end": -1,
                "pose_range": {       # 初始位姿范围
                    "z": (0.12, 0.42),          # 高度 12~42cm
                    "roll": (-0.1, 0.1),        # 姿态小扰动
                    "pitch": (-0.2, 0.2),
                    "yaw": (-torch.pi, torch.pi),  # 朝向随机
                },
                "velocity_range": {   # 初始速度范围（下落的初速度）
                    "x": (-2.5, 2.5),
                    "y": (-0.5, 0.5),
                    "z": (-0., 0.),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.5, 0.5),
                },
                "command_limits": {   # 落地后前 3 秒的指令限制（别太难）
                    "duration_s": 3.0,
                    "lin_vel_x": (-2.5, 2.5),
                    "lin_vel_y": (0.0, 0.0),
                    "ang_vel_z": (-0.5 * torch.pi, 0.5 * torch.pi),
                    "height": (0.18, 0.43),
                },
                "leg_length_range": (0.15, 0.35),   # 初始腿长范围
                "leg_angle_range": (-0.25 * torch.pi, 0.25 * torch.pi),  # 初始腿摆角
            }
        ),
    }

    def __post_init__(self):
        super().__post_init__()
        # —— 核心改动：切到相对高度口径 + 打开腾空状态机 ——
        self.use_absolute_height = False      # 改用相对地面高度（要开扫描仪）
        self.enable_state_machines = True     # 状态机总开关打开
        self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
        self.airborne_state_machine_cfg["enabled"] = True   # ★打开腾空状态机
        _enable_v14_body_height_scanner(self)   # 开车身高度扫描仪
        _enable_v14_wheel_height_scanners(self) # 开轮子高度扫描仪（腾空判定要测轮离地）
        self.commands.special_modes['spin_low'].iteration_start = 0   # 自旋/冲刺提前开练
        self.commands.special_modes['spin_mid'].iteration_start = 0
        self.commands.special_modes['dash'].iteration_start = 0
        self.commands.special_modes['dash'].rel_envs = 0.2
        self.commands.special_modes["zero_cmd"] = mdp.SpecialModeEntryCfg(  # 新模式：完全零指令（纯练落地）
            rel_envs=0.1,                     # 10% 环境
            iteration_start=0,
            iteration_end=-1,
            disable_jump_takeoff=True,
            debug_print=False,
            ranges=mdp.SpecialModeEntryCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        )
        self.predefined_reset_ground["modes"]["positive"]["prob"] = 0.2   # 地面重置姿势概率微调
        self.predefined_reset_ground["modes"]["negative"]["prob"] = 0.1
        # —— 腾空状态的奖励覆盖 ——
        reward_scales = dict(self.airborne_state_machine_cfg.get("reward_scales", {}))
        reward_scales["undesired_contact"] = 25.0       # 腾空期乱接触罚×25
        reward_scales["flat_orientation_y_v"] = 0.0     # 腾空期俯仰角速度惩罚取消（空中本来要摆姿势）
        # reward_scales["flat_orientation_x_v"] = 0.0
        # reward_scales["flat_orientation_y_exp"] = 0.1
        reward_scales["termination"] = 6.0              # 腾空期摔倒罚 6（比平时 -200 温和）
        # reward_scales["lin_vel_z"] = 0.1
        # extra
        # reward_scales["leg_joint_acc"] = 0.1
        # reward_scales["leg_joint_vel"] = 0.1
        # reward_scales["track_height_exp_tight"] = 0.1
        reward_scales["track_height_square"] = 0.0      # 腾空期身高平方惩罚取消
        reward_scales["foot_bound_square"] = 0.0        # 腾空期脚甩惩罚取消
        self.airborne_state_machine_cfg["reward_scales"] = reward_scales
        reward_full = dict(self.airborne_state_machine_cfg.get("reward_full", {}))
        reward_full.update(                             # 满额奖励覆盖（当前全部注释）
            {
                # "track_lin_vel_xy": 0.8,
                # "track_lin_vel_xy_square": 0.0,
                # "track_ang_vel_z": 0.8,
                # "track_ang_vel_z_square": 0.0,
                # "track_height_exp_tight": 0.8,
                # "track_height_square": 0.0,
            }
        )
        self.airborne_state_machine_cfg["reward_full"] = reward_full
        reward_additions = dict(self.airborne_state_machine_cfg.get("reward_additions", {}))
        reward_additions.update(                        # ★追加 v1 专属附加奖励项定义
            {
                "airborne_undesired_contact_force": {   # 腾空期接触力 L1 惩罚
                    "type": "undesired_contact_force",
                    "force_threshold": self.undesired_contact_force_threshold,
                    "mode": "l1",
                },
                "airborne_landing_down_vel": {          # 落地后仍在下坠的惩罚（要缓冲住）
                    "type": "negative_lin_vel_z_after_wheel_contact",
                    "start_duration_s": 0.02,
                    "use_world_frame": True,
                    "mode": "l2",
                    "square_sigma": 2.0,
                },
                "airborne_landing_down_vel_exp": {      # 同上的 exp 版（软着陆奖励）
                    "type": "negative_lin_vel_z_after_wheel_contact_exp",
                    "start_duration_s": 0.02,
                    "use_world_frame": True,
                    "sigma": 0.25,
                },
                "airborne_joint_pos_limits": {          # 关节撞限位惩罚
                    "type": "rear2_rear1_joint_pos_limits",
                },
                "airborne_joint_pos_limits_vel_reg": {  # 限位处速度惩罚
                    "type": "rear2_rear1_joint_pos_limits_vel_reg",
                },
                "airborne_leg_length_min": {            # 空中腿收太短惩罚（要伸长准备着地）
                    "type": "leg_retraction",
                    "mode": "below_target_per_leg",
                    "target": 0.25,
                    "before_wheel_contact_duration_s": 0.02,
                    "contact_mode": "any_wheel",
                },
                "airborne_wheel_height_below_base": {   # 轮子要伸到车体下方 0.34m（准备接地的姿势）
                    "type": "wheel_height_below_base_exp",
                    # target 表示轮子底部相对 base 低多少米，不是轮心高度。
                    "target": 0.34,
                    "sigma": 0.025,
                    "before_wheel_contact_duration_s": 0.02,
                    "contact_mode": "any_wheel",
                },
                "airborne_wheel_heading_x_centering": { # 落地时轮子朝向要与前向对齐（别侧着砸）
                    "type": "wheel_heading_x_centering",
                    "wheel_contact_duration_s": 0.02,
                    "base_contact_duration_s": 0.02,
                    "wheel_heading_z_max": -0.1,
                    "sigma": 0.02,
                },
                # "airborne_wheel_height_below_base_tight": {  # （注释掉：更严格的版本）
                #     "type": "wheel_height_below_base_exp",
                #     # target 表示轮子底部相对 base 低多少米，不是轮心高度。
                #     "target": 0.22,
                #     "sigma": 0.005,
                #     "before_wheel_contact_duration_s": 0.02,
                #     "contact_mode": "any_wheel",
                # },
                "airborne_low_body_height": {           # 车身压得太低惩罚
                    "type": "body_height_below",
                    "threshold": 0.3,
                    "mode": "l2",
                    "square_sigma": 5.0,
                },
                "airborne_body_height_below_binary": {  # 车身低于阈值的一票否决惩罚
                    "type": "body_height_below",
                    "threshold": 0.25,
                    "mode": "binary",
                },
            }
        )
        self.airborne_state_machine_cfg["reward_additions"] = reward_additions
        # self.rewards['foot_bound_square'] = -1.0
        # —— 附加项的权重（只有配了权重才生效）——
        self.rewards['rear2_rear1_joint_pos_limits'] = 0.0        # 平时关（落地才需要）
        self.rewards['rear2_rear1_joint_pos_limits_torque'] = 0.0
        self.rewards['rear2_rear1_joint_pos_limits_vel'] = 0.0
        # self.rewards['airborne_undesired_contact_force'] = -1.0
        # self.rewards['airborne_landing_down_vel'] = -1.0
        # self.rewards['airborne_landing_down_vel_exp'] = 1.0
        self.rewards['airborne_joint_pos_limits'] = -10.0         # 关节撞限位 -10
        # self.rewards['airborne_joint_pos_limits_vel_reg'] = -100.0
        # self.rewards["airborne_leg_length_min"] = -40.0
        # self.rewards["airborne_wheel_height_below_base"] = 40.0
        self.rewards["airborne_wheel_heading_x_centering"] = 10.0 # 轮向对齐 +10（最重要）
        # self.rewards["airborne_wheel_height_below_base_tight"] = 10.0
        # self.rewards["airborne_low_body_height"] = -10.0
        # self.rewards["airborne_body_height_below_binary"] = -10.0

        # —— 动作/关节相关惩罚调轻（落地需要大幅动作，别罚太狠）——
        self.rewards["action_rate"] = -0.002
        self.rewards["action_smoothness_leg"] = -0.005
        self.rewards["action_smoothness_wheel"] = -0.001
        self.rewards["leg_joint_acc"] = -1e-7
        self.rewards["leg_joint_vel"] = -1e-3
        self.rewards["wheel_acc"] = -2e-9
        self.rewards["wheel_vel"] = -2e-6

        _apply_v14_airborne_landing_precontact_cfg(self)   # 最后套用共享的落地/预接触奖励配置


''' 小陀螺训练 '''
@configclass
class WheelbipeV14RoughEnvCfg(WheelbipeV14FlatEnvCfg_v2):
# class WheelbipeV14RoughEnvCfg(WheelbipeV14FlatEnvCfg):
    # ★粗糙地形 + 自旋训练：在 _v2（航向锁定+小陀螺平移）基础上换粗糙地形
    # play_keep_done_reset = True
    rough_terrain_generator_cfg = copy.deepcopy(mdp.RM_ROTATION_TERRAINS_CFG_99)  # 粗糙地形生成器（旋转场景专用组合）
    rough_terrain_command_overrides_cfg = copy.deepcopy(V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_1)  # 各地形上的指令覆盖表
    # rough_height_offset_curriculum_cfg = {   # （注释掉：地形难度课程，需要时打开）
    #     "enabled": True,
    #     "interval": 400,          # 每 400 轮升一级
    #     "max_iteration": 5000,    # 5000 轮后封顶
    #     "num_levels": 11,         # 共 11 级难度
    #     "steps_per_iteration": 24,
    #     "random_reset_up_to_current_level": False,
    #     "random_reset_after_max": True,      # 毕业后全等级随机
    #     "randomize_type_on_random_reset": True,
    # }
    rough_terrain_boundary_reset_cfg = {     # 跑出地形边界就按超时重置
        "enabled": True,
        "margin": 0.5,            # 边界余量 0.5m
        "use_inner_terrain_area": False,   # 算上外围缓冲带
    }
    predefined_reset_air = {                 # 空中重置模式（当前 enabled=False 关闭，仅保留定义）
        "enabled": False,
        "modes": (
            {
                "name": "air_1",             # 模式1：较低高度扔下
                "prob": 0.2,
                "iteration_start": 0,
                "iteration_end": -1,
                "pose_range": {
                    "z": (0.05, 0.25),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-torch.pi, torch.pi),
                },
                "velocity_range": {
                    "x": (-2.0, 2.0),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.0, 0.0),
                },
                "leg_length_range": (0.20, 0.35),
                "leg_angle_range": (-0.25 * torch.pi, 0.25 * torch.pi),
            },
            {
                "name": "air_2",             # 模式2：较高高度扔下（第 2000 轮后启用）
                "prob": 0.2,
                "iteration_start": 2000,
                "iteration_end": -1,
                "pose_range": {
                    "z": (0.25, 0.35),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-torch.pi, torch.pi),
                },
                "velocity_range": {
                    "x": (-2.0, 2.0),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
                "leg_length_range": (0.20, 0.35),
                "leg_angle_range": (-0.25 * torch.pi, 0.25 * torch.pi),
            },
        ),
    }

    vel_orientation_y_gate_enabled: bool = False   # 粗糙地形上姿态本来会晃，门控都关掉
    vel_height_gate_enabled: bool = False          # （不平的地面上身高误差大，罚了会误导）

    def __post_init__(self):
        super().__post_init__()
        # self.predefined_reset_ground['start_root_height'] = 0.25
        # self.predefined_reset_ground['prob'] = 0.3
        _apply_v14_rough_runtime_cfg(self)    # 套用粗糙地形运行时配置（cfg_utils：换地形、指令覆盖等）
        self.enable_state_machines = False    # 关状态机
        self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
        self.airborne_state_machine_cfg["enabled"] = False
        self.wheel_forward_scan_cfg = copy.deepcopy(self.wheel_forward_scan_cfg)
        self.wheel_forward_scan_cfg["enabled"] = False
        # _disable_v14_wheel_height_scanners(self)
        # self.height_range = [0.2,0.4]
        self.commands.special_modes['spin_low'].iteration_start = 0    # 自旋/冲刺提前开练
        # self.commands.special_modes['spin_low'].rel_envs = 0.3
        self.commands.special_modes['spin_mid'].iteration_start = 0
        # self.commands.special_modes['spin_mid'].rel_envs = 0.3
        self.commands.special_modes['dash'].iteration_start = 0
        # self.commands.special_modes['dash'].rel_envs = 0.
        # self.commands.special_modes["zero_cmd"] = mdp.SpecialModeEntryCfg(   # （注释掉：零指令模式）
        #     rel_envs=0.1,
        #     iteration_start=0,
        #     iteration_end=-1,
        #     disable_jump_takeoff=True,
        #     debug_print=False,
        #     ranges=mdp.SpecialModeEntryCfg.Ranges(
        #         lin_vel_x=(0.0, 0.0),
        #         lin_vel_y=(0.0, 0.0),
        #         ang_vel_z=(0.0, 0.0),
        #     ),
        # )
        self.rewards['wheel_power'] = -1e-5   # 粗糙地形上轮功率惩罚调轻（需要更多动力爬坡）
        self.rewards['joint_torque'] = -1e-5  # 力矩惩罚同理调轻
        self.stand_still_deadzone_enabled = True
        self.rewards["stand_still_lin_vel"] = -1.0

''' 跑场训练 '''
@configclass
class WheelbipeV14RoughEnvCfg_v1(WheelbipeV14FlatEnvCfg_v1):
    # ★粗糙地形跑场训练：在腾空预训练(_v1)基础上换粗糙地形 + 跑得更快
    rough_terrain_boundary_reset_cfg = {     # 越界重置打开
        "enabled": True,
        "margin": 0.5,
        "use_inner_terrain_area": False,
    }
    def __post_init__(self):
        super().__post_init__()
        # self.predefined_reset_ground['prob'] = 0.3
        self.predefined_reset_air = copy.deepcopy(self.predefined_reset_air)
        self.predefined_reset_air["enabled"] = True   # 打开空中重置（结合 _v1 的落地能力）
        _apply_v14_rough_runtime_cfg(self)    # 套用粗糙地形运行时配置
        # self.commands.resampling_time_range = (3.0, 7.0)
        self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
        terrain_command_resample = copy.deepcopy(     # 腾空期间重采指令打开
            self.airborne_state_machine_cfg.get("terrain_command_resample", {})
        )
        terrain_command_resample["enabled"] = True
        terrain_command_resample["lin_vel_x_sign_from_current"] = True   # x 速度方向跟随原指令
        self.airborne_state_machine_cfg["terrain_command_resample"] = terrain_command_resample
        self.rewards['wheel_power'] = -1e-5   # 动力惩罚调轻（跑场要多出力）
        self.rewards['joint_torque'] = -1e-5
        self.rewards['track_lin_vel_xy'] = 1.25   # 速度追踪权重加码（跑得准更重要）

@configclass
class WheelbipeV14FlatEnvCfg_Play(WheelbipeV14FlatEnvCfg):
    # 平地基础任务的演示配置
    events = EventCfgV14_Play()       # Play 事件表
    curriculum = None                 # 无课程
    use_frame_stack = False
    num_obs_hist = 1
    num_privileged_obs_hist = 1

    def __post_init__(self):
        super().__post_init__()
        # self.episode_length_s = 2.0
        # self.episode_length_s = 3.0
        # self.height_range = [0.25, 0.25]
        # self.predefined_reset_ground["modes"]["positive"]["prob"] = 0.
        # self.predefined_reset_ground["modes"]["negative"]["prob"] = 0.
        # self.predefined_reset_air = {    # （注释掉：演示用的固定空中重置方案）
        #     "enabled": True,
        #     "modes": (
        #         {
        #             "name": "play_air",
        #             "prob": 1.0,
        #             "iteration_start": 0,
        #             "iteration_end": -1,
        #             "pose_range": {
        #                 "z": (0.35, 0.35),
        #                 "roll": (-0.05, 0.05),
        #                 "pitch": (-0.1, 0.1),
        #                 "yaw": (-torch.pi, torch.pi),
        #             },
        #             "velocity_range": {
        #                 "x": (2., 2.),
        #                 "y": (-0.25, 0.25),
        #                 "z": (0.0, 0.0),
        #                 "roll": (0.0, 0.0),
        #                 "pitch": (0.0, 0.0),
        #                 "yaw": (-0.0, 0.0),
        #             },
        #             "leg_length_range": (0.20, 0.30),
        #             "leg_angle_range": (-0.2 * torch.pi, 0.2 * torch.pi),
        #         },
        #     ),
        # }

        self.play = True              # 标记 Play 模式
        self.play_height_scanner_debug_vis = True   # 可视化高度扫描仪
        self.play_terrain_debug_vis = True          # 可视化地形信息
        self.play_wheel_motor_z_axis_align_debug = True   # 打印轮轴对齐调试信息
        self.play_wheel_motor_z_axis_align_debug_interval = 50
        self.play_wheel_motor_z_axis_align_debug_env_id = 0
        self.play_wheel_material_debug = False      # 轮摩擦打印关闭
        self.play_wheel_material_debug_interval = 50
        self.play_wheel_material_debug_env_id = 0

# ─────────────────────────────────────────────────────────────────────────────
# 以下 6 组是"算法变体"配置：环境本身不变，只改观测空间的组织方式，
# 以匹配 DreamWaQ / HIMLoco / NP3O 三种算法对观测的要求（历史堆叠、特权观测、约束等）。
# 维度常量（V14_BASE_POLICY_OBS_DIM 等）都定义在 cfg_utils.py 里。
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class WheelbipeV14FlatDreamWaqEnvCfg(WheelbipeV14FlatEnvCfg):
    """V14 flat experiment: DreamWaQ CENet policy with proprioceptive history."""

    # DreamWaQ：用"本体感知历史"训练 CENet 编码器隐式想象地形（无需真实高度扫描）
    # Disable base Flat's 7D ctrl_mode_obs so obs matches the declared 28/71
    # dims (otherwise obs_history init 28D vs appended 35D -> torch.stack crash).
    ctrl_mode_obs_enabled = False    # 关 7 维模式观测（否则维度与算法声明对不上会崩溃）
    # curriculum = CurriculumCfgV14()
    curriculum = None                # 无课程
    use_frame_stack = False
    num_obs_hist = V14_DREAMWAQ_POLICY_HIST        # 策略观测历史长度（DreamWaQ 需要长历史）
    num_privileged_obs_hist = 1
    n_state_est = V14_DREAMWAQ_ESTIMATED_STATE_DIM # 状态估计器（CENet）输出的隐状态维度
    observation_space = {            # DreamWaQ 的分组观测空间
        "policy": V14_BASE_POLICY_OBS_DIM,                       # 策略当前观测
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_DREAMWAQ_POLICY_HIST,  # 策略历史堆叠
        "critic": V14_BASE_PRIVILEGED_OBS_DIM,                   # critic 特权观测
        "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,              # 上一帧 critic 观测
        "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,              # critic 历史
    }
    state_space = V14_BASE_PRIVILEGED_OBS_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM            # 单帧维度对齐
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_BASE_PRIVILEGED_OBS_DIM
        self.observation_space = {                               # 再次显式设置（保证一致）
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_DREAMWAQ_POLICY_HIST,
            "critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
        }







@configclass
class WheelbipeV14FlatDreamWaqEnvCfg_Play(WheelbipeV14FlatEnvCfg_Play):
    """Play config matching the V14 flat DreamWaQ experiment."""

    # DreamWaQ 的演示配置（观测空间与训练版完全一致，仅切换 Play 行为）
    ctrl_mode_obs_enabled = False
    use_frame_stack = False
    num_obs_hist = V14_DREAMWAQ_POLICY_HIST
    num_privileged_obs_hist = 1
    n_state_est = V14_DREAMWAQ_ESTIMATED_STATE_DIM
    observation_space = {            # 同训练版
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_DREAMWAQ_POLICY_HIST,
        "critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
    }
    state_space = V14_BASE_PRIVILEGED_OBS_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_BASE_PRIVILEGED_OBS_DIM
        self.observation_space = {   # 同训练版
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_DREAMWAQ_POLICY_HIST,
            "critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
        }


@configclass
class WheelbipeV14FlatHIMEnvCfg(WheelbipeV14FlatEnvCfg):
    """V14 flat experiment: HIMLoco hybrid internal model with policy history."""

    # HIMLoco：用历史观测训练估计器替代显式地形编码器（混合内部模型）
    ctrl_mode_obs_enabled = False
    curriculum = CurriculumCfgV14()  # HIM 用了课程学习（身高奖励渐进）
    use_frame_stack = False
    num_obs_hist = V14_HIM_POLICY_HIST            # 策略历史长度（HIM 规定的帧数）
    num_privileged_obs_hist = 1
    n_state_est = V14_HIM_ESTIMATED_STATE_DIM     # HIM 估计器输出维度
    observation_space = {            # HIM 的分组观测空间（结构同 DreamWaQ，历史长度不同）
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_HIM_POLICY_HIST,
        "critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
    }
    state_space = V14_BASE_PRIVILEGED_OBS_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_BASE_PRIVILEGED_OBS_DIM
        self.observation_space = {   # 同上
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_HIM_POLICY_HIST,
            "critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
        }





@configclass
class WheelbipeV14FlatHIMEnvCfg_Play(WheelbipeV14FlatEnvCfg_Play):
    """Play config matching the V14 flat HIMLoco experiment."""

    # HIMLoco 的演示配置（观测空间与训练版一致）
    ctrl_mode_obs_enabled = False
    use_frame_stack = False
    num_obs_hist = V14_HIM_POLICY_HIST
    num_privileged_obs_hist = 1
    n_state_est = V14_HIM_ESTIMATED_STATE_DIM
    observation_space = {            # 同训练版
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_HIM_POLICY_HIST,
        "critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
        "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
    }
    state_space = V14_BASE_PRIVILEGED_OBS_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_BASE_PRIVILEGED_OBS_DIM
        self.observation_space = {   # 同训练版
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_HIM_POLICY_HIST,
            "critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "prev_critic": V14_BASE_PRIVILEGED_OBS_DIM,
            "critic_hist": V14_BASE_PRIVILEGED_OBS_DIM,
        }


@configclass
class WheelbipeV14FlatNP3OBarlowEnvCfg(WheelbipeV14FlatEnvCfg):
    """V14 flat experiment: NP3O cost channels with Barlow Twins history actor."""

    # NP3O：PPO + 安全约束(cost channels) + BarlowTwins 自监督历史编码
    ctrl_mode_obs_enabled = False
    # curriculum = CurriculumCfgV14()
    curriculum = None
    np3o_barlow_enabled = True       # 打开 BarlowTwins 分支
    use_frame_stack = False
    num_obs_hist = V14_NP3O_POLICY_HIST           # 策略历史长度
    num_privileged_obs_hist = 1
    n_scan = 0                       # 无高度扫描输入
    n_state_est = V14_NP3O_EST_DIM   # 特权隐变量维度
    n_priv_latent = V14_NP3O_EST_DIM # 特权编码器输出维度
    num_costs = V14_NP3O_COST_DIM    # 安全约束条数（倾角/身高/角速度/力矩/关节速度）
    np3o_cost_d_values = [0.0, 0.0, 0.0, 0.0, 0.0]     # 各约束的目标值
    np3o_cost_k_initial = [1.0, 1.0, 1.0, 0.5, 0.5]    # 各约束的初始拉格朗日乘子
    np3o_tilt_limit_deg = 20.0       # 约束1：车身倾角 ≤20°
    np3o_body_height_min = 0.18      # 约束2：身高 ≥0.18m
    np3o_body_height_max = 0.42      #         ≤0.42m
    np3o_ang_vel_xy_limit = 4.0      # 约束3：横滚/俯仰角速度 ≤4
    np3o_torque_limit = 30.0         # 约束4：力矩 ≤30
    np3o_joint_velocity_limit = 80.0 # 约束5：关节速度 ≤80
    np3o_cost_clip = 100.0           # 约束值裁剪上限
    vel_height_gate_enabled = True   # 开身高门控
    vel_orientation_y_gate_enabled = False
    observation_space = {            # NP3O 的分组观测空间（多了约束观测）
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST,
        "priv_latent": V14_NP3O_EST_DIM,             # 特权隐变量
        "on_constraint": V14_NP3O_ON_CONSTRAINT_DIM, # 约束观测
    }
    state_space = V14_NP3O_ON_CONSTRAINT_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_NP3O_ON_CONSTRAINT_DIM
        self.observation_space = {   # 同上
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST,
            "priv_latent": V14_NP3O_EST_DIM,
            "on_constraint": V14_NP3O_ON_CONSTRAINT_DIM,
        }
        # self.rewards = copy.deepcopy(self.rewards)   # （注释掉：NP3O 专用的奖励权重覆盖方案）
        # self.rewards.update(
        #     {
        #         "flat_orientation_y": -0.0,
        #         "flat_orientation_y_v": -1.0,
        #         "flat_orientation_x": -0.0,
        #         "flat_orientation_x_v": -0.5,
        #         "ang_vel_xy": -0.002,
        #         "lin_vel_z": -0.2,
        #         "joint_torque": -2.0e-5,
        #         "leg_joint_vel": -1.0e-3,
        #     }
        # )




@configclass
class WheelbipeV14FlatNP3OBarlowEnvCfg_Play(WheelbipeV14FlatEnvCfg_Play):
    """Play config matching the V14 flat NP3O + Barlow Twins experiment."""

    # NP3O 的演示配置（演示时倾角约束收紧到 15°）
    ctrl_mode_obs_enabled = False
    np3o_barlow_enabled = True
    use_frame_stack = False
    num_obs_hist = V14_NP3O_POLICY_HIST
    num_privileged_obs_hist = 1
    n_scan = 0
    n_state_est = V14_NP3O_EST_DIM
    n_priv_latent = V14_NP3O_EST_DIM
    num_costs = V14_NP3O_COST_DIM
    np3o_cost_d_values = [0.0, 0.0, 0.0, 0.0, 0.0]
    np3o_cost_k_initial = [1.0, 1.0, 1.0, 0.5, 0.5]
    np3o_tilt_limit_deg = 15.0       # 演示时倾角约束收到 15°（更稳）
    np3o_body_height_min = 0.18
    np3o_body_height_max = 0.42
    np3o_ang_vel_xy_limit = 4.0
    np3o_torque_limit = 30.0
    np3o_joint_velocity_limit = 80.0
    np3o_cost_clip = 100.0
    observation_space = {            # 同训练版
        "policy": V14_BASE_POLICY_OBS_DIM,
        "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST,
        "priv_latent": V14_NP3O_EST_DIM,
        "on_constraint": V14_NP3O_ON_CONSTRAINT_DIM,
    }
    state_space = V14_NP3O_ON_CONSTRAINT_DIM

    def __post_init__(self):
        super().__post_init__()
        self.num_single_obs = V14_BASE_POLICY_OBS_DIM
        self.num_single_privileged_obs = V14_BASE_PRIVILEGED_OBS_DIM
        self.state_space = V14_NP3O_ON_CONSTRAINT_DIM
        self.observation_space = {   # 同训练版
            "policy": V14_BASE_POLICY_OBS_DIM,
            "policy_hist": V14_BASE_POLICY_OBS_DIM * V14_NP3O_POLICY_HIST,
            "priv_latent": V14_NP3O_EST_DIM,
            "on_constraint": V14_NP3O_ON_CONSTRAINT_DIM,
        }

@configclass
class WheelbipeV14RoughEnvCfg_Play(WheelbipeV14RoughEnvCfg):
    # ★粗糙地形小陀螺任务的演示配置：固定飞坡地形 + 自动录轨迹
    rough_height_offset_curriculum_cfg = {   # 课程关闭（演示固定地形）
        **V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG,
        "enabled": False,
    }
    rough_terrain_boundary_reset_cfg = {     # 越界重置（用内部区域判定）
        "enabled": True,
        "margin": 0.5,
        "use_inner_terrain_area": True,
    }

    curriculum = None
    use_frame_stack = False
    num_obs_hist = 1
    num_privileged_obs_hist = 1
    episode_length_s = 5.0           # 每局 5 秒
    events = EventCfgV14_Play()      # Play 事件表

    def __post_init__(self):
        super().__post_init__()
        play_terrain_name = "cliff_inv_stair_slope_short_for_rm_play"  # 演示地形名（断崖反台阶坡）
        self.predefined_reset_ground['prob'] = 0.0   # 不从地面出生
        self.predefined_reset_air = {                # 统一从空中"飞入"出生（模拟冲坡腾空）
            "enabled": True,
            "modes": (
                {
                    "name": "play_air",
                    "prob": 1.0,                     # 100% 走这个模式
                    "iteration_start": 0,
                    "iteration_end": -1,
                    "pose_range": {
                        "z": (0.25, 0.25),           # 固定 25cm 高
                        "roll": (-0.05, 0.05),
                        "pitch": (-0.1, 0.1),
                        "yaw": (-torch.pi, torch.pi),
                    },
                    "velocity_range": {
                        "x": (1.9, 2.0),             # 固定前冲 1.9~2.0 m/s
                        "y": (-0.25, 0.25),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (-0.0, 0.0),
                    },
                    "leg_length_range": (0.20, 0.35),
                    "leg_angle_range": (-0.2 * torch.pi, 0.2 * torch.pi),
                },
            ),
        }
        self.episode_length_s = 5.       # 每局 5 秒（同上）
        _play_terrain_gen = copy.deepcopy(mdp.RM_ROUGH_TERRAINS_PLAY_CFG)  # Play 专用地形生成器
        if len(getattr(_play_terrain_gen, "sub_terrains", {}) or {}) == 0:
            _play_terrain_gen.sub_terrains = {       # 生成器为空时手动塞入演示地形
                play_terrain_name: mdp.CLIFF_INV_STAIR_SLOPE_SHORT_FOR_RM_PLAY,
            }
        _play_terrain_gen.num_rows = 10              # 10 行难度等级
        _play_terrain_gen.curriculum = True          # 行=难度递增
        self.terrain_command_overrides = _filter_v14_terrain_command_overrides(  # 只保留该地形需要的指令覆盖
            V14_ROUGH_TERRAIN_COMMAND_OVERRIDES, _play_terrain_gen
        )
        self.terrain = TerrainImporterCfg(           # 地面配置
            prim_path="/World/ground",               # 挂载路径
            terrain_type="generator",                # 程序化生成地形
            collision_group=-1,
            terrain_generator=_play_terrain_gen,     # 用上面的生成器
            physics_material=sim_utils.RigidBodyMaterialCfg(  # 地面摩擦材质
                friction_combine_mode="multiply",    # 与轮子摩擦相乘
                restitution_combine_mode="multiply",
                static_friction=1.0,                 # 静摩擦 1.0
                dynamic_friction=1.0,                # 动摩擦 1.0
                restitution=0.0,                     # 无弹性
            ),
            debug_vis=False,
        )
        # self.height_range = [0.25, 0.35]
        self.events.robot_joint_stiffness_and_damping = None   # 演示时不随机化增益（固定手感）
        self.commands = mdp.UniformVelocityCommandCfg(   # 换成普通命令生成器（演示不需要特殊模式）
            asset_name="robot",
            resampling_time_range=(5, 15),       # 5~15 秒重采
            rel_standing_envs=0.0,               # 不站住
            rel_heading_envs=0.5,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=True,                      # 画出指令箭头（演示用）
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(2.2, 2.2),            # 固定 2.2 m/s 前冲（飞坡）
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-torch.pi, torch.pi),
                heading=(-torch.pi, torch.pi),
            ),
        )
        self.play = False                 # 注意：这里 play=False（靠下面的固定指令/地形演示飞坡）
        self.play_height_scanner_debug_vis = True
        self.play_terrain_debug_vis = True
        self.velocity_trace_cfg = {       # ★打开速度轨迹录制（CSV+HTML）
            "enabled": True,
            "terrain_name": play_terrain_name,   # 在该地形上挑 agent 录
            "agent_index": None,
            "lock_agent": True,              # 锁定同一个 agent
            "sample_dt": 0.02,               # 每 0.02s 采一帧
            "html_update_interval_s": 1.0,   # HTML 每秒刷新
            "max_rows": 20000,
            "unique_path": True,             # 文件名带时间戳防覆盖
            "csv_path": "logs/debug/rough_play_velocity_trace.csv",
            "html_path": "logs/debug/rough_play_velocity_trace.html",
        }

@configclass
class WheelbipeV14RoughEnvCfg_v1_Play(WheelbipeV14RoughEnvCfg_v1):
    # ★粗糙地形跑场任务(v1)的演示配置：同样固定地形 + 轨迹录制（输出到 logs/debug/730/）
    ctrl_mode_obs_enabled = True
    def __post_init__(self):
        super().__post_init__()
        play_terrain_name = "cliff_inv_stair_slope_short_for_rm_play"  # 演示地形
        self.episode_length_s = 5    # 每局 5 秒
        self.height_range = [0.25,0.25]   # 身高固定 0.25
        self.play = True             # Play 模式
        self.play_ang_vel_z_debug_vis = False
        self.play_height_scanner_debug_vis = True
        self.play_terrain_debug_vis = True
        self.velocity_trace_cfg = {  # 轨迹录制配置
            "enabled": True,
            "terrain_name": play_terrain_name,
            "agent_index": None,
            "lock_agent": True,
            "sample_dt": 0.02,
            "html_update_interval_s": 1.0,
            "max_rows": 20000,
            "unique_path": True,
            "csv_path": "logs/debug/730/rough_v1_play_velocity_trace.csv",
            "html_path": "logs/debug/730/rough_v1_play_velocity_trace.html",
        }
        _play_terrain_gen = copy.deepcopy(mdp.RM_ROUGH_TERRAINS_PLAY_CFG)  # Play 地形生成器
        _play_terrain_gen.curriculum = True
        self.terrain_command_overrides = _filter_v14_terrain_command_overrides(  # 过滤指令覆盖
            V14_ROUGH_TERRAIN_COMMAND_OVERRIDES, _play_terrain_gen
        )
        self.terrain = TerrainImporterCfg(   # 地面配置（同上）
            prim_path="/World/ground",
            terrain_type="generator",
            collision_group=-1,
            terrain_generator=_play_terrain_gen,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
