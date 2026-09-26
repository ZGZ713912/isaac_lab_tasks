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
#   ③ 当前阶段固定低车身 q=Q_LOW，只训练低车身主动悬挂
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
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass

from agent_tasks.direct.deformable_suspension import cfg_utils as du
from agent_world.assets.deformable_V2 import DeformableInfantryCFG
from agent_world.terrains import HfCustomPeriodicSlopeTerrainCfg


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
    # 必须与 PPO num_steps_per_env 一致，否则 base_contact_death 等按 iteration
    # 计数的课程会提前/滞后激活（曾导致 iteration≈500 全灭）。
    training_progress_steps_per_iteration = 48

    # ---- 动作 / 腿级联 PID（固定低车身主动悬挂训练）----
    leg_action_scale = 0.10  # rad per action unit
    # Phase-0 实测（scripts/tools/deformable_sign_probe.py）：Q_LOW 时底盘余量约 7mm，
    # q≈1.096 时剩 4.9mm，q≈1.136 起底盘开始承载（16N→203N）。因此有效安全上限
    # 是 1.13 而不是关节限位 1.36。放开到这个值以取得最大下行行程，同时用
    # terminate_chassis_clearance + chassis_ground 兜底。
    leg_target_upper_limit = 1.13
    use_leg_cascade_pid = True  # False 回退单环位置 PD（对比/调试）
    leg_vel_cmd_limit = 4.0  # rad/s，外环速度指令限幅（< 资产 velocity_limit 17）
    leg_outer_kp = 6.0  # 1/s：位置误差 → 速度指令
    # 斜坡上 ki=0 会在重力分量下产生稳态下压 → base 长期触地；保留小积分抗静差。
    leg_outer_ki = 0.5  # 1/s²
    leg_inner_kp = 2.0  # N·m/(rad/s)：速度误差 → 力矩
    leg_inner_ki = 2.0  # N·m/rad
    leg_outer_int_limit = 0.4  # rad·s：外环积分限幅（抗积分饱和）
    leg_inner_int_limit = 0.5  # rad：内环积分限幅（抗积分饱和）
    max_leg_torque = 13.0  # 训练力矩终限幅
    # 单环位置 PD 回退参数（use_leg_cascade_pid=False 时使用）
    leg_stiffness = 50.0
    leg_damping = 5.0

    # ---- 基准车高（当前阶段固定低车身，不训练高度切换）----
    use_continuous_q_cmd = False  # False=每次 reset 从 q_cmd_choices 采样
    q_cmd_choices = (du.Q_LOW,)
    q_cmd_range = (du.Q_LOW, du.Q_LOW)
    default_q_cmd = du.Q_LOW
    init_root_height = 0.18  # spawn 时 base 原点离地高度（略高于接触，轻微下落）
    reset_height_buffer = 0.03  # reset 时的小高度缓冲，避免把出生冲击学成控制策略
    low_mode_q_threshold = 0.5 * (du.Q_HIGH + du.Q_LOW)  # q_cmd 高于此角视为“低模式”（q 大=车低）

    # ---- 第一阶段只训练静态主动悬挂，运动伺服后置 ----
    enable_chassis_servo = False
    chassis_total_mass = 25.5  # 整车质量粗估（仅伺服力标定用）
    chassis_yaw_inertia = 1.0  # 偏航惯量粗估
    chassis_servo_kp_lin = 20.0  # 1/s（速度误差 → 加速度）
    chassis_servo_kp_yaw = 10.0  # 1/s
    chassis_servo_max_force = 300.0  # N（按轴限幅；训练时再受 μ·N_total 牵引限幅约束）
    chassis_servo_max_torque = 150.0  # N·m（训练时再受 μ·N_total·L 约束）
    chassis_servo_friction_coeff = 0.6  # μ：地面可传递牵引力上限 |F| ≤ μ·N_total
    # True=施加库仑牵引帽（训练默认）；False=play 只保留绝对 max_force/torque。
    chassis_servo_friction_cap_enabled = True
    airborne_force_threshold = 5.0  # N：四轮法向力之和低于此视为悬空（仅用于日志）

    # ---- 第一阶段无运动命令；先学会低车身静态调平 ----
    cmd_lin_vel_x_range = (0.0, 0.0)
    cmd_lin_vel_y_range = (0.0, 0.0)
    cmd_ang_vel_z_range = (0.0, 0.0)
    cmd_resample_time_range = (3.0, 5.0)
    cmd_rel_standing_envs = 0.0

    # ---- 方向均匀性：分层/循环 spawn（车体系坡度方位 + 上/下坡相位）----
    spawn_dir_stratify = True  # True=按 bin 循环精确均匀覆盖各朝向
    spawn_dir_bins = 8  # 车体系上坡梯度方位 bin 数（需能整除 batch 更均匀）
    spawn_dir_jitter = True  # bin 内均匀抖动，避免过拟合到轴上
    spawn_phase_stratify = True  # 上/下坡出生相位分层

    # ---- 越界重置：跑出单位格（env_spacing）即按 time_out 重置 ----
    boundary_reset_margin = 0.5  # 距单元边界的余量（m）
    boundary_reset_enabled = True  # 键盘 play 等场景可关闭

    # ---- 外部命令覆盖：True 时不做命令重采样，cmd_buf 完全由外部（键盘）写入 ----
    external_cmd_override = False

    # ---- 底盘触地两阶段：前 N 轮软惩罚（不死亡），之后死亡 ----
    # 用正确 steps/iteration 后，1000 ≈ 真实 PPO iteration 1000；再留裕量防过早全灭。
    base_contact_death_after_iterations = 2000


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
    wheel_contact_force_threshold = 1.0  # 单轮接地判定阈值（N）；不表示目标载荷
    undesired_contact_force_threshold = 3.0  # 腿/轮架触地惩罚阈值（N）

    # ---- 奖励形状参数 ----
    orientation_x_exp_sigma = 0.05  # roll（pgb_y）
    orientation_y_exp_sigma = 0.05  # pitch（pgb_x）
    q_track_sigma = 0.02  # 基准角跟踪 σ (rad²)
    low_height_sigma = 3.0e-4  # 低模式贴地偏好 σ (m²)
    # IMU 重力水平分量 -> 每腿 q 修正。
    # Phase-0 实测符号：front_raise -> pgb_x<0，left_raise -> pgb_y<0；
    # 故 pgb_x>0（前低）须抬前 -> sign=-1。原 +1 会让前低时抬后腿（方向相反）。
    tilt_leg_q_sign = -1.0
    tilt_leg_q_gain = 0.5  # rad per normalized projected-gravity component（抬/压双向）
    tilt_leg_position_sigma = 0.5  # rad²，腿角二次误差归一化尺度（原 0.02 放大 50 倍是爆点来源）

    # ---- 底盘离地保护（Phase-0 实测，强惩罚 + 终止）----
    chassis_ground_threshold = 0.006  # m：低于此余量开始软惩罚
    chassis_ground_scale = 0.006  # m：惩罚归一化尺度（余量 0 时项=1）
    terminate_chassis_clearance = 0.0  # m：底盘余量低于此值判死（负值=允许轻微穿透）

    # ---- 奖励清洗与限幅（防 1e4~1e5 爆点）----
    reward_term_clip = 100.0
    reward_total_clip = 1000.0

    # ---- 奖励权重（低车身主动悬挂；不约束轮间载荷转移）----
    rewards = OrderedDict(
        alive=0.02,
        termination=-200.0,
        all_wheel_contact=2.5,
        tilt_leg_position_error=-0.1,
        tilt_quadratic=-12.0,
        flat_orientation_x_exp=1.5,
        flat_orientation_y_exp=1.5,
        track_q_cmd_exp=0.02,
        low_height_pref=0.0,
        chassis_ground=-20.0,
        torques=-1.0e-4,
        action_rate=-0.02,
        action_rate2=0.0,
        leg_joint_vel=-5.0e-3,
        leg_joint_acc=0.0,
        leg_torque_rate=0.0,
        base_ang_acc=0.0,
        base_lin_acc_z=0.0,
        ang_vel_xy=-0.03,
        lin_vel_z=-0.05,
        undesired_contact=-10.0,
        base_contact=-10.0,
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

    # 先验证固定低车身的调平奖励和执行器，域随机化后置。
    events = None

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


def _make_periodic_slope_terrain(
    angle_range: tuple[float, float],
    seed: int = 0,
    size: tuple[float, float] = (150.0, 150.0),
) -> TerrainImporterCfg:
    """共享大平面周期坡面地形：单 tile，剖面沿 x，每周期独立随机坡角。"""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        use_terrain_origins=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
        terrain_generator=TerrainGeneratorCfg(
            size=size,
            border_width=2.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            difficulty_range=(0.5, 0.5),
            sub_terrains={
                "periodic_slope": HfCustomPeriodicSlopeTerrainCfg(
                    proportion=1.0,
                    segment_length=1.0,
                    angle_range=angle_range,
                    angle_seed=seed,
                ),
            },
        ),
    )


@configclass
class DeformableSuspensionRoughEnvCfg(DeformableSuspensionBaseEnvCfg):
    """周期坡面连续跨越地形（课程第一段）：+θ(1m) → 平(1m) → −θ(1m) → 平(1m)（周期 4m）。

    - 每周期独立随机 θ ∈ [10,17]°（第二段 17–25°，见 Steep 子类），上下坡对称；
    - 共享大平面：单 tile，env 按 env_spacing 网格铺在同一张面上（env 出生网格在 env 内
      整体偏移到地形 footprint 内）；
    - 出生按方位 bin 分层循环（车体系坡度方向）+ 相位分层（上/下坡），10cm 下落；
      越界按 time_out 重置。
    - 底盘触地两阶段：<1000 轮软惩罚，之后死亡。
    """

    # 当前阶段固定低车身；坡面训练后置，不恢复高度切换。
    q_cmd_choices = (du.Q_LOW,)
    default_q_cmd = du.Q_LOW

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=6.0, replicate_physics=True
    )

    terrain = _make_periodic_slope_terrain(angle_range=(10.0, 17.0), seed=0)
    termination_roll_deg = 45.0
    termination_pitch_deg = 45.0


@configclass
class DeformableSuspensionRoughSteepEnvCfg(DeformableSuspensionRoughEnvCfg):
    """课程第二段：坡度 17–25°，放宽姿态终止（允许“尽力而为”）。"""

    terrain = _make_periodic_slope_terrain(angle_range=(17.0, 25.0), seed=0)
    termination_roll_deg = 60.0
    termination_pitch_deg = 60.0


@configclass
class DeformableSuspensionFlatPlayEnvCfg(DeformableSuspensionFlatEnvCfg):
    play: bool = True


@configclass
class DeformableSuspensionRoughPlayEnvCfg(DeformableSuspensionRoughEnvCfg):
    play: bool = True


@configclass
class DeformableSuspensionRoughSteepPlayEnvCfg(DeformableSuspensionRoughSteepEnvCfg):
    play: bool = True


@configclass
class DeformableSuspensionRoughKeyboardPlayEnvCfg(DeformableSuspensionRoughEnvCfg):
    """键盘遥控 play（第一段地形 10–17°）。

    单 env、底盘伺服开、命令完全由键盘写入（不做随机重采样）、关越界/终止/域随机化。
    关分层 spawn：手动观察时用随机朝向与相位。
    """

    play: bool = True
    spawn_dir_stratify = False
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=6.0, replicate_physics=True
    )
    episode_length_s = 1.0e6  # 不因超时重置（键盘观察）
    external_cmd_override = True
    enable_chassis_servo = True
    cmd_lin_vel_x_range = (0.0, 0.0)
    cmd_lin_vel_y_range = (0.0, 0.0)
    cmd_ang_vel_z_range = (0.0, 0.0)
    cmd_resample_time_range = (1.0e9, 1.0e9)
    boundary_reset_enabled = False
    events = None  # 关域随机化，play 时确定性
    termination_roll_deg = 90.0  # sin(90°)=1 → 实际不触发
    termination_pitch_deg = 90.0
    terminate_base_height_low = -1.0e9
    base_contact_death_after_iterations = 1_000_000_000
    # 键盘 play 专用伺服增益（克服球轮静摩擦，仅影响 play；训练用基类默认值）
    chassis_servo_kp_lin = 80.0
    chassis_servo_max_force = 1600.0
    # yaw 惯量用真值量级；旧值 15 会把自旋指令算大再被摩擦帽压死 → Z/X 极慢。
    chassis_yaw_inertia = 1.5
    chassis_servo_kp_yaw = 30.0
    chassis_servo_max_torque = 900.0
    # play 不设摩擦帽：只保留绝对 max_force/torque，移动/自旋跟手。
    chassis_servo_friction_cap_enabled = False


@configclass
class DeformableSuspensionRoughSteepKeyboardPlayEnvCfg(DeformableSuspensionRoughSteepEnvCfg):
    """键盘遥控 play（第二段地形 17–25°）。"""

    play: bool = True
    spawn_dir_stratify = False
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=6.0, replicate_physics=True
    )
    episode_length_s = 1.0e6
    external_cmd_override = True
    enable_chassis_servo = True
    cmd_lin_vel_x_range = (0.0, 0.0)
    cmd_lin_vel_y_range = (0.0, 0.0)
    cmd_ang_vel_z_range = (0.0, 0.0)
    cmd_resample_time_range = (1.0e9, 1.0e9)
    boundary_reset_enabled = False
    events = None
    termination_roll_deg = 90.0
    termination_pitch_deg = 90.0
    terminate_base_height_low = -1.0e9
    base_contact_death_after_iterations = 1_000_000_000
    # 键盘 play 专用伺服增益（同上）
    chassis_servo_kp_lin = 80.0
    chassis_servo_max_force = 1600.0
    chassis_yaw_inertia = 1.5
    chassis_servo_kp_yaw = 30.0
    chassis_servo_max_torque = 900.0
    chassis_servo_friction_cap_enabled = False
