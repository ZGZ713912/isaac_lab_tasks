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
    motion_curriculum_iterations = 2000

    action_contract_version = "minangle_residual_v2"
    leg_min_physical_angle = du.PHYSICAL_MIN_ANGLE
    q_cmd_choices = (du.Q_MINANGLE,)
    q_cmd_range = (du.Q_MINANGLE, du.Q_MINANGLE)
    default_q_cmd = du.Q_MINANGLE
    leg_action_scale = 0.35  # inherited V0 hook only; V2 uses full-stroke signed spans
    leg_target_upper_limit = du.Q_LOW
    leg_target_rate_limit = 2.0  # rad/s, separate from the ADRC tracking differentiator
    # RMCS deformable-infantry-omni-rl.yaml; no normalized q_max/span scaling.
    leg_max_physical_angle = math.radians(75.0)
    adrc_dt = 0.001
    # Simulation input-gain calibration; RMCS raw b0=-1 remains an explicit ablation.
    adrc_b0 = -10.0
    adrc_feedback_applied_torque = True
    adrc_kt = 1.0
    adrc_td_h = 0.001
    adrc_td_r = 50.0
    adrc_td_max_vel = float("inf")
    adrc_td_max_acc = float("inf")
    adrc_eso_w0 = 250.0
    adrc_z3_limit = 1.0e9
    adrc_k1 = 30.0
    adrc_k2 = 17.0
    adrc_alpha1 = 0.75
    adrc_alpha2 = 0.7
    adrc_delta = 0.02
    adrc_u_min = -200.0
    adrc_u_max = 200.0
    adrc_output_min = -200.0
    adrc_output_max = 200.0
    max_leg_torque = 25.0
    use_leg_cascade_pid = False  # V1 uses ADRC directly; neither legacy branch is executed.
    decimation = 10
    sim = DeformableSuspensionFlatEnvCfg().sim.copy()
    sim.dt = adrc_dt
    sim.render_interval = decimation
    contact_sensor = DeformableSuspensionFlatEnvCfg().contact_sensor.copy()
    contact_sensor.update_period = adrc_dt
    drive_dynamics_version = "low_slip_v2"
    wheel_velocity_kp = 0.8
    wheel_velocity_ki = 2.0  # near-critical damping with the 26.34 kg chassis load
    drive_linear_acceleration_limit = 2.0  # m/s^2, in the command frame
    drive_yaw_acceleration_limit = 4.0  # rad/s^2
    # Allow the continuous modulation required by simultaneous world translation/yaw.
    wheel_acceleration_limit = 200.0  # rad/s^2; secondary actuator bound
    wheel_axial_inertia = 0.002092387  # kg m^2; deformable_V2 URDF wheel ixx
    wheel_torque_limit = 5.0
    wheel_speed_limit = 60.0
    # Low longitudinal slip under normal loads, integrated at 1 kHz. This stiffness
    # and the wheel PI gains are tuned together for the URDF wheel inertia.
    tire_slip_stiffness = 500.0  # N/(m/s)
    tire_contact_stiffness = 5000.0  # N/m; elastic contact supports static slope loads
    tire_lateral_drag = 0.2  # passive omni rollers have low transverse resistance
    tire_friction_range = (0.8, 1.0)
    tire_contact_gap = 0.003
    wheel_contact_force_threshold = 3.0

    max_body_top_height = 0.255
    enforce_tunnel_height = False
    height_termination_margin = 0.025  # warmup training permits transient overshoot, logs enforce 255 mm
    height_settle_steps = 50
    reset_height_buffer = 0.01
    height_penalty_weight = 2.0
    baseline_reward_weight = 0.3
    reward_scale = 0.1
    horizontal_tolerance_deg = 3.0
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
    rewards["all_wheel_contact"] = 5.0
    rewards["wheel_load_balance"] = 0.5
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=6.0, replicate_physics=True)
    robot_cfg = DeformableSuspensionFlatEnvCfg().robot_cfg.copy()
    robot_cfg.actuators["legs"].effort_limit_sim = max_leg_torque
    robot_cfg.actuators["legs"].effort_limit = None
    robot_cfg.actuators["wheels"].damping = 0.0
    robot_cfg.actuators["wheels"].effort_limit_sim = 5.0
    robot_cfg.actuators["wheels"].velocity_limit_sim = 60.0
    robot_cfg.spawn.rigid_props.max_angular_velocity = 7200.0  # USD uses degrees/s, including wheel spin
    robot_cfg.actuators["wheels"].friction = 0.0
    robot_cfg.actuators["wheels"].dynamic_friction = 0.0
    robot_cfg.actuators["wheels"].viscous_friction = 0.0


@configclass
class DeformableDynamicRoughEnvCfg(DeformableDynamicFlatEnvCfg):
    terrain = _make_periodic_slope_terrain(angle_range=(0.0, 5.0), seed=0)


@configclass
class DeformableDynamicRoughSteepEnvCfg(DeformableDynamicRoughEnvCfg):
    terrain = _make_periodic_slope_terrain(angle_range=(5.0, 8.0), seed=1)


@configclass
class DeformableBestEffortEnvCfg(DeformableDynamicRoughEnvCfg):
    """Stage one: learn continuous correction before adding impossible slopes."""
    best_effort_leveling = True
    baseline_reward_weight = 0.02
    # Contact and clearance outweigh any benefit from unloading a wheel to level.
    rewards = OrderedDict(DeformableDynamicRoughEnvCfg().rewards)
    rewards["all_wheel_contact"] = 8.0
    rewards["wheel_load_balance"] = 0.0
    best_effort_tilt_weight = 6.0  # radians, active at every tilt, not cut off at 10 deg
    termination_roll_deg = 45.0
    termination_pitch_deg = 45.0


@configclass
class DeformableBestEffortTenDegreeEnvCfg(DeformableBestEffortEnvCfg):
    terrain = _make_periodic_slope_terrain(angle_range=(5.0, 10.0), seed=2)


@configclass
class DeformableBestEffortPrecisionEnvCfg(DeformableBestEffortEnvCfg):
    """Foundation precision trial: distinguish small tilt while retaining contact."""
    orientation_x_exp_sigma = 0.004
    orientation_y_exp_sigma = 0.004


@configclass
class DeformableBestEffortOverloadEnvCfg(DeformableBestEffortEnvCfg):
    terrain = _make_periodic_slope_terrain(angle_range=(10.0, 15.0), seed=3)


@configclass
class DeformableDynamicKeyboardPlayEnvCfg(DeformableDynamicRoughEnvCfg):
    play = True
    external_cmd_override = True
    commands_world_frame = False
    spawn_dir_stratify = False
    boundary_reset_enabled = False
    episode_length_s = 120.0
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=6.0, replicate_physics=True)


def _legacy_config(cfg):
    cfg.action_contract_version = "legacy_v1"
    cfg.q_cmd_choices = (du.Q_LOW,)
    cfg.q_cmd_range = (du.Q_LOW, du.Q_LOW)
    cfg.default_q_cmd = du.Q_LOW
    cfg.motion_curriculum_iterations = 4000
    cfg.enforce_tunnel_height = True
    cfg.reward_scale = 1.0
    cfg.rewards["all_wheel_contact"] = 2.5
    cfg.rewards.pop("wheel_load_balance", None)


@configclass
class DeformableLegacyFlatEnvCfg(DeformableDynamicFlatEnvCfg):
    def __post_init__(self):
        _legacy_config(self)


@configclass
class DeformableLegacyRoughEnvCfg(DeformableDynamicRoughEnvCfg):
    terrain = _make_periodic_slope_terrain(angle_range=(2.0, 8.0), seed=0)

    def __post_init__(self):
        _legacy_config(self)


@configclass
class DeformableLegacySteepEnvCfg(DeformableDynamicRoughSteepEnvCfg):
    terrain = _make_periodic_slope_terrain(angle_range=(8.0, 17.0), seed=1)

    def __post_init__(self):
        _legacy_config(self)


@configclass
class DeformableLegacyKeyboardEnvCfg(DeformableDynamicKeyboardPlayEnvCfg):
    terrain = _make_periodic_slope_terrain(angle_range=(2.0, 8.0), seed=0)

    def __post_init__(self):
        _legacy_config(self)
