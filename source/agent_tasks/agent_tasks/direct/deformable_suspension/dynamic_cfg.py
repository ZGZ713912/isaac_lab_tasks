"""V1 wheel-contact dynamics; the same physical limits apply in training and play."""

import math
from collections import OrderedDict

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import cfg_utils as du
from .env_cfg import DeformableSuspensionFlatEnvCfg, _make_periodic_slope_terrain


@configclass
class DeformableDynamicFlatEnvCfg(DeformableSuspensionFlatEnvCfg):
    policy_history_length = 8
    observation_space = 32 * policy_history_length
    state_space = 40
    enable_chassis_servo = False
    events = None
    commands_world_frame = True
    cmd_lin_vel_x_range = (-1.0, 1.0)
    cmd_lin_vel_y_range = (-1.0, 1.0)
    cmd_ang_vel_z_range = (-2.0 * math.pi, 2.0 * math.pi)
    cmd_resample_time_range = (2.0, 4.0)
    motion_curriculum_iterations = 4000

    # q increases toward the low pose. V1 actions request extension away from it.
    leg_extension_range = du.Q_LOW
    leg_target_upper_limit = du.Q_LOW
    leg_outer_ki = 3.0
    leg_inner_ki = 20.0
    leg_inner_int_limit = 0.65
    leg_nominal_load = 25.5 * 9.81 / 4.0
    wheel_velocity_kp = 0.12
    wheel_torque_limit = 5.0
    wheel_speed_limit = 60.0
    tire_slip_stiffness = 12.0  # N/(m/s); stable at 200 Hz with the small wheel inertia
    tire_lateral_drag = 0.2  # passive omni rollers have low transverse resistance
    tire_friction_range = (0.5, 0.9)
    tire_contact_gap = 0.003
    wheel_contact_force_threshold = 3.0

    max_body_top_height = 0.255
    height_termination_margin = 0.025  # warmup training permits transient overshoot, logs enforce 255 mm
    height_settle_steps = 50
    reset_height_buffer = 0.01
    height_penalty_weight = 2.0
    baseline_reward_weight = 0.3
    base_contact_death_after_iterations = 0
    terminate_body_top = None  # V1 uses highest wheel's local ground, not root ground

    max_sensor_delay_steps = 2
    encoder_noise_std = 0.15  # rad/s
    encoder_bias_std = 0.05
    gyro_noise_std = 0.015  # rad/s
    gyro_bias_std = 0.01
    gravity_noise_std = 0.003
    rewards = OrderedDict(DeformableSuspensionFlatEnvCfg().rewards)
    rewards["tilt_leg_position_error"] = 0.0
    rewards["track_q_cmd_exp"] = 0.0
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=6.0, replicate_physics=True)
    robot_cfg = DeformableSuspensionFlatEnvCfg().robot_cfg.copy()
    robot_cfg.actuators["wheels"].damping = 0.0
    robot_cfg.actuators["wheels"].effort_limit_sim = 5.0
    robot_cfg.actuators["wheels"].velocity_limit_sim = 60.0
    robot_cfg.spawn.rigid_props.max_angular_velocity = 7200.0  # USD uses degrees/s, including wheel spin
    robot_cfg.actuators["wheels"].friction = 0.0
    robot_cfg.actuators["wheels"].dynamic_friction = 0.0
    robot_cfg.actuators["wheels"].viscous_friction = 0.0


@configclass
class DeformableDynamicRoughEnvCfg(DeformableDynamicFlatEnvCfg):
    terrain = _make_periodic_slope_terrain(angle_range=(2.0, 8.0), seed=0)


@configclass
class DeformableDynamicRoughSteepEnvCfg(DeformableDynamicRoughEnvCfg):
    terrain = _make_periodic_slope_terrain(angle_range=(8.0, 17.0), seed=1)


@configclass
class DeformableDynamicKeyboardPlayEnvCfg(DeformableDynamicRoughEnvCfg):
    play = True
    external_cmd_override = True
    commands_world_frame = False
    spawn_dir_stratify = False
    boundary_reset_enabled = False
    episode_length_s = 120.0
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=True)
