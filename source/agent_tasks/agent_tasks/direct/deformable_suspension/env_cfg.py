# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# DeformableSuspension —— 变形底盘主动悬挂任务（Isaac Lab 移植版）
#
# 与 RMCS rmcs_rl 部署合同逐项同构（sim-to-real by construction）：
#   obs 22: cmd3 | height_cmd1 | ang_vel3 | gravity3 | leg_pos4 | leg_vel4 | act4
#   act  4: 腿关节位置 PD 目标（action_scale 0.25，kp=200 kd=4）
# 轮子零驱动（自由滚动）；leg 与 wheel_set 用虚拟刚弹簧模拟平四耦合。
# 控制频率 100Hz（decimation 2 × dt 0.005）== 部署 rl_inference_frequency。
# =============================================================================

from collections import OrderedDict

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import (
    PyramidSlopedTerrainCfg,
    RandomRoughTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass

from agent_world.assets.deformable_suspension import DeformableSuspensionCFG


@configclass
class DeformableSuspensionBaseEnvCfg(DirectRLEnvCfg):
    """基础配置：合同、动力学、奖励、场景。"""

    # env
    decimation = 2  # 100 Hz 控制（与部署 rl_inference_frequency 一致）
    episode_length_s = 20.0
    action_space = 4  # 4 腿位置 PD 目标
    observation_space = 22  # cmd3|height1|angvel3|grav3|pos4|vel4|act4
    state_space = 26  # 特权观测：+ lin_vel3 + 车体相对高度1（asymmetric critic）
    play: bool = False

    # ---- 任务参数（与部署 rmcs_rl 配置严格对齐）----
    leg_action_scale = 0.25  # rad per action unit
    default_height_cmd = 0.132  # 名义站姿高度指令（对应 q=1.3439 = 77°，水平夹角系）
    height_range = (0.05, 0.17)  # 高度指令范围（三档基准：普通 0.132 / 高 0.16+ / 低 0.05~0.06）
    init_root_height = 0.18  # 初始车体原点离地高度（名义 0.132 + 缓冲，防初始穿透）
    termination_roll_deg = 35.0
    termination_pitch_deg = 35.0
    terminate_base_height_low = 0.03  # 接触力终止兜底，高度终止仅防数值异常
    leg_stiffness = 200.0  # 与部署 position_kp 一致
    leg_damping = 4.0  # 与部署 position_kd 一致
    max_leg_torque = 40.0  # 训练力矩限幅（部署 position_torque_max 按标定收紧）
    coupling_stiffness = 1000.0  # 平四虚拟弹簧
    coupling_damping = 10.0
    undesired_contact_force_threshold = 3.0  # 腿/轮架触地惩罚阈值
    desired_contact_force_threshold = 5.0  # 四轮着地奖励参考力
    orientation_x_exp_sigma = 0.02  # 水平奖励 σ（roll，pgb_y）
    orientation_y_exp_sigma = 0.04  # 水平奖励 σ（pitch，pgb_x）
    height_track_sigma = 0.005  # 高度跟踪奖励 σ

    # ---- 观测缩放（与部署 obs_*_scale 一致）----
    height_scale = 5.0
    ang_vel_scale = 0.5
    joint_pos_scale = 1.0
    joint_vel_scale = 0.1

    # ---- 奖励权重 ----
    rewards = OrderedDict(
        alive=1.0,
        termination=-100.0,
        flat_orientation_y_exp=2.0,
        flat_orientation_x_exp=2.0,
        four_wheel_contact=5.0,
        track_height_exp=1.5,
        torques=-1.0e-4,
        action_rate=-0.01,
        leg_joint_vel=-5.0e-3,
        leg_joint_acc=-5.0e-7,
        ang_vel_xy=-0.05,
        lin_vel_z=-0.5,
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

    # ---- robot ----
    robot_cfg: ArticulationCfg = DeformableSuspensionCFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )


@configclass
class DeformableSuspensionFlatEnvCfg(DeformableSuspensionBaseEnvCfg):
    """平面地形（调试/快速验证）。"""

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
    """粗糙地形（斜坡 + 随机粗糙，30%/70%），高度指令全范围训练。"""

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
            num_rows=10,  # 难度行（v1 固定难度，curriculum 留待后续）
            num_cols=20,
            difficulty_range=(0.4, 1.0),
            sub_terrains={
                "pyramid_sloped": PyramidSlopedTerrainCfg(
                    proportion=0.3, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
                ),
                "random_rough": RandomRoughTerrainCfg(
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
