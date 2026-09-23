# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V2 平地 DirectRLEnv 配置。
#
# 架构移植自 V40 训练仓 wheeled-biped-rl-train/src/wheeled_tasks/direct/v40_serial/env_cfg.py：
# 环境的具体维度/时钟/资产在 WheelLegV2Env.__init__ 里按合同（contract）填充。
# 合同动作和关节状态使用真实电机树关节：髋 L/R、膝 LL/RR、轮 L/R；
# 闭链输出由 USD 中的 spherical 约束传递，不再把主链膝关节当作电机输入。
# =============================================================================

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RigidBodyMaterialCfg, SimulationCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from agent_world.terrains import HfCustomPeriodicSlopeTerrainCfg

from .contract import DEFAULT_CONTRACT, ROUND2_CONTRACT


@configclass
class WheelLegV2EnvCfg(DirectRLEnvCfg):
    """Wheel_leg_V2 平地任务的基类；子类只改 stage。"""

    decimation = 2
    episode_length_s = 20.0
    is_finite_horizon = False
    # [L_joint1, LL_joint1, L_joint3, R_joint1, RR_joint1, R_joint3]
    action_space = 6
    # DirectRLEnv 2.3 会把 observation_space 包成 "policy"，state_space 包成 "critic"。
    observation_space = 125
    state_space = 29
    # 轮腿课程学习：当前轮次 = common_step_counter // iteration_steps + iteration_offset。
    # iteration_steps 必须等于 PPO 的 num_steps_per_env；resume 续训时把已训轮次写进 offset。
    iteration_steps: int = 48
    iteration_offset: int = 0
    # 接触摩擦：包胶轮用 realistic μ；默认 0.5 时单轮轮端 ~2.0 Nm 就超摩擦打滑。
    sim = SimulationCfg(
        dt=0.005,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9),
    )
    # 实测（/tmp 探测脚本）：
    #   replicate_physics=True + clone_in_fabric=False → 闭链 loop joints 被正确复制，
    #     多环境闭合误差 <0.001mm；这是可扩到 4096 env 的关键。
    #   clone_in_fabric=True → 接触传感器初始化失败（Failed to initialize contact
    #     reporter），必须保持 False。
    scene = InteractiveSceneCfg(
        num_envs=256, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False,
    )

    # 以下字段在 DirectRLEnv 构造场景之前由 __init__ 按合同填充
    robot_cfg: ArticulationCfg | None = None
    contact_sensor_cfg: ContactSensorCfg | None = None
    contract_path: str | None = str(DEFAULT_CONTRACT)
    stage: str = "stand"

    # 闭链误差看门狗（米）：任一闭合点两杆世界点距离超过该值即安全 reset；0 关闭。
    closure_error_tolerance: float = 0.005

    # 气弹簧力学：对 LLL/RRR 棱柱约束两端的刚体施加轴向力（BKB0.45-063-172, 10MPa）。
    # 几何（杆件/锚点）来自 constraints.json 的 gas_springs；这里只放力曲线参数。
    # 方向：气簧越压缩弹力越大 —— 109mm 全压 347N > 172mm 全伸 279N。
    # 数值取自产品图 9/12MPa 曲线数字插值到 10MPa（行程 0=279N, 行程 63.5=347N）；
    # 若实机充气压力不是 10MPa，或曲线/型号改动，需按图重新标定。
    gas_spring_enabled: bool = True
    gas_spring_force_at_min_n: float = 347.0   # 最短（109mm，全压缩）时的弹力
    gas_spring_force_at_max_n: float = 279.0   # 最长（172mm，全伸出）时的弹力
    gas_spring_damping_n_s_per_m: float = 0.0  # 阻尼未实测，默认 0

    # 地形：None = 平地（默认）；子类可挂 TerrainImporterCfg（周期坡面）。
    terrain: TerrainImporterCfg | None = None
    # 周期坡面 spawn：按坡角对齐 pitch 后，整车再抬高该值落地。
    slope_spawn_drop_m: float = 0.04

    # 跳跃（wheelbipe 语义）：仅 stage == "jump" 时生效。
    jump_enabled: bool = True
    jump_trigger_rate_per_s: float = 0.5
    jump_min_episode_time_s: float = 3.0
    jump_cooldown_s: float = 1.5
    jump_peak_height_range: tuple[float, float] = (0.35, 0.45)
    jump_push_start_height_m: float = 0.23
    jump_release_height_m: float = 0.29
    jump_assist_prob_start: float = 1.0
    jump_assist_prob_end: float = 0.0
    jump_assist_decay_iterations: int = 400
    jump_steps_per_iteration: int = 48
    jump_assist_force_z: float = 0.0
    jump_assist_missing_vel_gain: float = 200.0
    jump_assist_max_force_z: float = 400.0

    # 评估/演示：固定 (vx, wz, height) 覆盖采样命令；None = 正常训练采样。
    evaluation_command: tuple[float, float, float] | None = None


@configclass
class WheelLegV2StandEnvCfg(WheelLegV2EnvCfg):
    stage = "stand"


@configclass
class WheelLegV2HeightEnvCfg(WheelLegV2EnvCfg):
    stage = "height"


@configclass
class WheelLegV2FlatEnvCfg(WheelLegV2EnvCfg):
    stage = "locomotion"


@configclass
class WheelLegV2FlatRound2EnvCfg(WheelLegV2FlatEnvCfg):
    """V40 完成 profile：观测噪声 + 持续倾倒终止 + reset 根速度随机。"""
    contract_path = str(ROUND2_CONTRACT)


@configclass
class WheelLegV2FlatRound2PlayEnvCfg(WheelLegV2FlatRound2EnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False)


@configclass
class WheelLegV2StandPlayEnvCfg(WheelLegV2StandEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False)


@configclass
class WheelLegV2FlatPlayEnvCfg(WheelLegV2FlatEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False)


def _make_periodic_slope_terrain(
    angle_range: tuple[float, float],
    seed: int = 0,
    size: tuple[float, float] = (150.0, 150.0),
) -> TerrainImporterCfg:
    """共享大平面周期坡面地形：单 tile，剖面沿 x，每周期独立随机坡角（同 deformable）。"""
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
class WheelLegV2SlopeEnvCfg(WheelLegV2FlatRound2EnvCfg):
    """斜坡课程第一段：周期坡面 10–17°（round2 profile），环境 6m 间距铺在同一张大坡面上。"""

    scene = InteractiveSceneCfg(num_envs=256, env_spacing=6.0, replicate_physics=True, clone_in_fabric=False)
    terrain = _make_periodic_slope_terrain(angle_range=(10.0, 17.0), seed=0)


@configclass
class WheelLegV2SlopeSteepEnvCfg(WheelLegV2SlopeEnvCfg):
    """斜坡课程第二段：坡度 17–25°。"""

    terrain = _make_periodic_slope_terrain(angle_range=(17.0, 25.0), seed=0)


@configclass
class WheelLegV2SlopePlayEnvCfg(WheelLegV2SlopeEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=True, clone_in_fabric=False)


@configclass
class WheelLegV2SlopeSteepPlayEnvCfg(WheelLegV2SlopeSteepEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=True, clone_in_fabric=False)


@configclass
class WheelLegV2JumpEnvCfg(WheelLegV2FlatRound2EnvCfg):
    """原地/行进跳跃（wheelbipe 语义，round2 profile）：随机触发 + 弹道参考 + 辅助力。"""

    stage = "jump"
    scene = InteractiveSceneCfg(num_envs=256, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False)


@configclass
class WheelLegV2JumpPlayEnvCfg(WheelLegV2JumpEnvCfg):
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False)
