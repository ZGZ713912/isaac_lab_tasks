# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# DeformableSuspension —— 平四变形底盘主动悬挂任务（单层 DirectRLEnv，照 wheelbipe 惯例）
#
# 任务：用 4 个平四腿电机控制 4 个轮子的高度，使车体在
#   ① 四轮贴地且法向载荷尽量平均（不打滑）
#   ② base_link 尽量/严格水平
#   ③ 跟踪两档基准车高（高 q=0 / 低 q=1.254），低档软偏好低车高（隧道 260mm）
# 目标下工作。轮子为球体碰撞、无牵引力，故“运动”由外部底盘速度伺服实现（首版静态关闭）。
#
# 合同：
#   policy obs 26 = q_cmd1 | cmd3(vx,vy,ωz) | ang_vel_b3 | proj_grav_b3
#                  | leg_pos4(绝对角) | leg_vel4 | leg_torque4 | act4
#   critic    34 = policy 26 + lin_vel_b3 + 真实车高1 + 四轮接触力4
#   act        4 = joint_leg_* 位置 PD 目标（手工 effort PD，kp=200/kd=4）
# 控制频率 100Hz（decimation 2 × dt 1/200）。
#
# 备注：腿位置观测用**绝对关节角**（不用相对 q−q_cmd）——平四 q→车高非线性，
# 同 Δq 在不同基准下轮子抬升量不同，相对量会丢信息。
# =============================================================================

from collections import OrderedDict

import agent_tasks.manager.mdp.isaaclab as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import (
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass

from agent_tasks.direct.deformable_suspension import cfg_utils as du
from agent_world.assets.deformable_V2 import DeformableInfantryCFG


@configclass
class EventCfg:
    """域随机化事件表（首版静态两档不启用运行；v2 接 EventManager 时直接挂上）。

    写法照 wheel_leg_v1/wheelbipe：func 取 manager/mdp/isaaclab 的函数式包装，
    params 用 (min,max) 范围，asset_cfg 指向机器人的 body/joint 子集。
    注意：DirectRLEnv 不会自动建 EventManager，需在 env 内手工构造后才会生效。
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg_1", "leg_2", "leg_3", "leg_4",
                                     "upper_leg_1", "upper_leg_2", "upper_leg_3", "upper_leg_4"]
            ),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    add_wheel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["wheel_set_1", "wheel_set_2", "wheel_set_3", "wheel_set_4"]
            ),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )


@configclass
class DeformableSuspensionBaseEnvCfg(DirectRLEnvCfg):
    """基础配置：合同、动力学、奖励、场景（平坦/粗糙/Play 变体只覆盖 terrain 与少量参数）。"""

    # ---- env ----
    decimation = 2  # 100 Hz 控制
    episode_length_s = 20.0
    action_space = 4
    observation_space = 26  # q_cmd1|cmd3|angvel3|grav3|pos4|vel4|torque4|act4
    state_space = 34  # critic：+ lin_vel3 + 车高1 + 四轮接触力4
    play: bool = False
    training_progress_steps_per_iteration = 24

    # ---- 动作 / PD（与部署 rmcs_rl 同构）----
    leg_action_scale = 0.25  # rad per action unit
    leg_stiffness = 200.0  # 部署 position_kp
    leg_damping = 4.0  # 部署 position_kd
    max_leg_torque = 40.0  # 训练力矩限幅

    # ---- 基准车高（两档 q_cmd；首版离散两档，q_cmd 以连续量进 obs）----
    use_continuous_q_cmd = False  # False=每次 reset 从 q_cmd_choices 采样
    q_cmd_choices = (du.Q_HIGH, du.Q_LOW)  # 高=初始 0°；低≈1cm 离地
    q_cmd_range = (du.Q_HIGH, du.Q_LOW)
    default_q_cmd = du.Q_HIGH
    init_root_height = 0.18  # spawn 时 base 原点离地高度（略高于接触，轻微下落）
    reset_height_buffer = 0.02  # reset 时在 q_to_height(q_cmd) 之上留的缓冲
    low_mode_q_threshold = 0.5 * (du.Q_HIGH + du.Q_LOW)  # q_cmd 高于此角视为“低模式”（q 大=车低）

    # ---- 外部底盘速度伺服（球体轮无牵引力；首版静态关闭）----
    enable_chassis_servo = False
    chassis_total_mass = 25.5  # 整车质量粗估（仅伺服力标定用）
    chassis_yaw_inertia = 1.0  # 偏航惯量粗估
    chassis_servo_kp_lin = 20.0  # 1/s（速度误差 → 加速度）
    chassis_servo_kp_yaw = 10.0  # 1/s
    chassis_servo_max_force = 300.0  # N
    chassis_servo_max_torque = 150.0  # N·m

    # ---- 运动命令（首版全 0 = 静态；v2 打开伺服并给范围）----
    cmd_lin_vel_x_range = (0.0, 0.0)
    cmd_lin_vel_y_range = (0.0, 0.0)
    cmd_ang_vel_z_range = (0.0, 0.0)
    cmd_resample_time_range = (3.0, 5.0)
    cmd_rel_standing_envs = 0.0

    # ---- 观测缩放 ----
    q_cmd_scale = 1.0
    cmd_scale = 1.0
    ang_vel_scale = 0.5
    joint_pos_scale = 1.0
    joint_vel_scale = 0.1
    joint_torque_scale = 0.05

    # ---- 判定阈值 ----
    termination_roll_deg = 30.0
    termination_pitch_deg = 30.0
    terminate_base_height_low = 0.004  # 低于此离地高度即终止（防穿透/塌）
    terminate_body_top: float | None = None  # 260mm 隧道约束（None=不启用）
    wheel_contact_force_threshold = 1.0  # 单轮着地判定（N）
    desired_contact_force_threshold = 20.0  # 均力参考力（N）
    undesired_contact_force_threshold = 3.0  # 腿/轮架触地惩罚阈值（N）

    # ---- 奖励形状参数 ----
    orientation_x_exp_sigma = 0.02  # roll（pgb_y）
    orientation_y_exp_sigma = 0.02  # pitch（pgb_x）
    wheel_force_balance_sigma = 50.0  # 均力 exp(-var/σ)，σ 单位 N²
    q_track_sigma = 0.02  # 基准角跟踪 σ (rad²)
    low_height_sigma = 3.0e-4  # 低模式贴地偏好 σ (m²)

    # ---- 奖励权重（v1；权重为每秒尺度，env 内不额外乘 step_dt）----
    rewards = OrderedDict(
        alive=1.0,
        termination=-200.0,
        four_wheel_contact=5.0,
        wheel_force_balance=4.0,
        flat_orientation_x_exp=2.0,
        flat_orientation_y_exp=2.0,
        track_q_cmd_exp=1.5,
        low_height_pref=1.0,
        torques=-1.0e-4,
        action_rate=-0.01,
        leg_joint_vel=-5.0e-3,
        leg_joint_acc=-5.0e-7,
        ang_vel_xy=-0.05,
        lin_vel_z=-0.2,
        undesired_contact=-10.0,
    )

    # ---- simulation ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ---- scene ----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=4.0, replicate_physics=True
    )

    # ---- robot / sensors ----
    robot_cfg: ArticulationCfg = DeformableInfantryCFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # ---- 事件 / 课程（v2 接入）----
    events = EventCfg()
    curriculum = None


@configclass
class DeformableSuspensionFlatEnvCfg(DeformableSuspensionBaseEnvCfg):
    """平面地形（首版训练/快速验证）。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


@configclass
class DeformableSuspensionRoughEnvCfg(DeformableSuspensionBaseEnvCfg):
    """粗糙地形（斜坡 + 随机粗糙 30%/70%）；课程/坡度渐进留待 v2。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            difficulty_range=(0.4, 1.0),
            sub_terrains={
                "pyramid_sloped": HfPyramidSlopedTerrainCfg(
                    proportion=0.3, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
                ),
                "random_rough": HfRandomUniformTerrainCfg(
                    proportion=0.7, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.25
                ),
            },
        ),
    )


@configclass
class DeformableSuspensionFlatPlayEnvCfg(DeformableSuspensionFlatEnvCfg):
    play: bool = True


@configclass
class DeformableSuspensionRoughPlayEnvCfg(DeformableSuspensionRoughEnvCfg):
    play: bool = True
