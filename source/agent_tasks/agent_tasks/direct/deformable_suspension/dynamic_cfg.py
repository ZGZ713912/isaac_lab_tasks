"""V1 wheel-contact dynamics; the same physical limits apply in training and play."""

import math
from collections import OrderedDict
from pathlib import Path

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
    soft_body_height_penalty = False
    height_termination_margin = 0.025  # warmup training permits transient overshoot, logs enforce 255 mm
    height_settle_steps = 50
    reset_height_buffer = 0.01
    height_penalty_weight = 2.0
    baseline_reward_weight = 0.3
    baseline_extension_penalty_weight = 0.0
    flat_baseline_extension_penalty_weight = 0.0
    low_profile_fade_grade_deg = 5.0
    training_steep_reference_mix = False
    steep_teacher_start_grade_deg = 17.0
    steep_teacher_full_grade_deg = 20.0
    drive_velocity_tracking_weight = 0.0
    drive_yaw_tracking_weight = 0.0
    static_traction_margin_weight = 0.0
    traction_reserve_fraction = 0.1
    traction_mass_kg = 26.3365312  # sum of the checked-in V2 URDF link masses
    clearance_margin_m = 0.0
    clearance_margin_weight = 0.0
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


class _DeformableReal2SimFields:
    """Current-domain actuator fields shared by train/evaluation variants.

    The source CSVs contain LK protocol counts.  ``set_joint_effort_target``
    still receives the explicit N*m result of the conversion in
    ``real2sim.py``; the legacy 0.174 Nm/count proxy is never used as a PhysX
    effort.
    """

    real2sim_enabled = True
    real2sim_model_path = str(Path(__file__).with_name("configs") / "deformable_real2sim.json")
    real2sim_current_limit = 2048.0
    # None means use the checked-in identification model.  Keep this overridable
    # for an explicit calibrated ablation without silently replacing the model.
    real2sim_torque_per_current_raw = None
    # Transport delay is an unmeasured prior, separate from the measured
    # 2 ms CAN period (timestamp delta / sequence delta in 5 ms CSV samples).
    real2sim_command_delay_steps_range = (0, 1)
    real2sim_command_period_steps_range = (2, 2)
    real2sim_feedback_delay_steps_range = (1, 2)
    real2sim_torque_scale_range = (0.90, 1.10)
    real2sim_torque_lag_tau_s = 0.003
    real2sim_current_noise_std_range = (4.0, 80.0)
    real2sim_angle_noise_std_range = (0.0004, 0.0025)
    real2sim_velocity_noise_std_range = (0.002, 0.02)
    real2sim_coulomb_friction_range = (0.03, 0.20)
    real2sim_viscous_friction_range = (0.005, 0.05)
    real2sim_backlash_range = (0.0, 0.01)
    real2sim_current_deadzone_raw = 1.0
    real2sim_backlash_engaged_scale = 0.25
    real2sim_friction_velocity_eps = 0.02
    real2sim_randomize = True
    real2sim_observation_version = "current_fraction_v1"
    # Match the RMCS current-domain controller and its observer input.
    adrc_output_domain = "current_raw"
    adrc_controller_output_to_current_raw = 5.74635241301908
    adrc_b0 = -1.0
    adrc_u_min = -2048.0
    adrc_u_max = 2048.0
    adrc_output_min = -2048.0
    adrc_output_max = 2048.0
    adrc_feedback_applied_torque = True
    max_leg_torque = 25.0


@configclass
class DeformableDynamicReal2SimEnvCfg(_DeformableReal2SimFields, DeformableDynamicFlatEnvCfg):
    """Flat current-domain real2sim training ablation."""


@configclass
class DeformableDynamicRoughReal2SimEnvCfg(_DeformableReal2SimFields, DeformableDynamicRoughEnvCfg):
    """0--5 degree rough current-domain real2sim training."""


@configclass
class DeformableBestEffortReal2SimEnvCfg(_DeformableReal2SimFields, DeformableBestEffortEnvCfg):
    """Best-effort leveling with measured current-domain timing/noise priors."""


@configclass
class DeformableBestEffortPrecisionReal2SimEnvCfg(
    _DeformableReal2SimFields, DeformableBestEffortPrecisionEnvCfg
):
    """Precision foundation task with the real2sim actuator boundary."""


@configclass
class DeformableReal2SimKeyboardPlayEnvCfg(_DeformableReal2SimFields, DeformableDynamicKeyboardPlayEnvCfg):
    """Fixed-parameter noisy GUI play; use ``--grade-deg=20`` for validation."""

    # A 20 degree validation ramp is intentionally outside the nominal 0--5
    # degree terrain range.  Start tangent to the local plane when four-wheel
    # horizontal contact is geometrically impossible, then let the policy act.
    best_effort_leveling = True
    baseline_reward_weight = 0.02
    best_effort_tilt_weight = 6.0
    termination_roll_deg = 45.0
    termination_pitch_deg = 45.0
    rewards = OrderedDict(DeformableDynamicKeyboardPlayEnvCfg().rewards)
    rewards["all_wheel_contact"] = 8.0
    rewards["wheel_load_balance"] = 0.0

    real2sim_randomize = False
    real2sim_command_delay_steps_range = (1, 1)
    real2sim_command_period_steps_range = (2, 2)
    real2sim_feedback_delay_steps_range = (1, 1)
    real2sim_torque_scale_range = (1.0, 1.0)
    real2sim_current_noise_std_range = (8.0, 8.0)
    real2sim_angle_noise_std_range = (0.0010, 0.0010)
    real2sim_velocity_noise_std_range = (0.008, 0.008)
    real2sim_coulomb_friction_range = (0.08, 0.08)
    real2sim_viscous_friction_range = (0.02, 0.02)
    real2sim_backlash_range = (0.004, 0.004)


class _DeformableFittedReal2SimFields(_DeformableReal2SimFields):
    """CSV-fitted assembly; explicit new angle/action and observation contract."""

    real2sim_model_path = str(Path(__file__).with_name("configs") / "deformable_real2sim_fitted.json")
    action_contract_version = "minangle_physical_v3"
    real2sim_observation_version = "current_fraction_v2"
    # CAD rod direction at q=0, distinct from the 75-degree command maximum.
    leg_physical_angle_zero = du.URDF_ZERO_PHYSICAL_ANGLE
    default_q_cmd = leg_physical_angle_zero - math.radians(17.)
    q_cmd_choices = (default_q_cmd,)
    q_cmd_range = (default_q_cmd, default_q_cmd)
    leg_target_lower_limit = leg_physical_angle_zero - math.radians(75.)
    # One degree below the soft baseline retains >6 mm nominal CAD clearance.
    leg_target_upper_limit = leg_physical_angle_zero - math.radians(16.)
    real2sim_command_delay_steps_range = (0, 0)
    # Fresh command rows see ~1.6ms-old feedback; the alternate control tick
    # sees a new packet. At 1kHz this requires odd feedback phase, not delay=0.
    real2sim_feedback_delay_steps_range = (1, 1)
    real2sim_feedback_period_steps_range = (2, 2)
    real2sim_torque_lag_tau_s = 0.0046
    real2sim_current_noise_std_range = (0., 0.)
    real2sim_current_sensor_noise_std_range = (2., 6.)
    real2sim_angle_noise_std_range = (0.00001, 0.0001)
    real2sim_velocity_noise_std_range = (0.0001, 0.002)
    # These are additional robustness variations around the fitted mechanism.
    real2sim_torque_scale_range = (0.90, 1.10)
    real2sim_coulomb_friction_range = (0., 0.)
    real2sim_viscous_friction_range = (0., 0.)
    real2sim_backlash_range = (0., 0.)  # not separately identifiable in these CSVs
    max_leg_torque = 0.021663417980643397 * 2048.
    real2sim_total_effort_limit_nm = 300.  # numerical bound also includes passive stop force
    robot_cfg = DeformableDynamicFlatEnvCfg().robot_cfg.copy()
    robot_cfg.actuators["legs"].effort_limit_sim = real2sim_total_effort_limit_nm


@configclass
class DeformableFittedFlatEnvCfg(_DeformableFittedReal2SimFields, DeformableDynamicFlatEnvCfg):
    """Flat CSV-fitted current-domain training."""


@configclass
class DeformableFittedRoughEnvCfg(_DeformableFittedReal2SimFields, DeformableDynamicRoughEnvCfg):
    """Rough CSV-fitted current-domain training."""


@configclass
class DeformableFittedPrecisionEnvCfg(_DeformableFittedReal2SimFields, DeformableBestEffortPrecisionEnvCfg):
    """Best-effort precision foundation with the fitted assembly."""


@configclass
class DeformableFittedAdaptiveEnvCfg(DeformableFittedPrecisionEnvCfg):
    """Mixed-grade optimization with a dense low-body preference.

    Longer ramps expose sustained grade correction as well as transitions.
    The fitted actuator, action/observation contracts and physical termination
    remain the same as the baseline experiment.
    """

    terrain = _make_periodic_slope_terrain(angle_range=(0.0, 20.0), seed=13)
    terrain.terrain_generator.sub_terrains["periodic_slope"].segment_length = 3.0
    scene = InteractiveSceneCfg(num_envs=256, env_spacing=8.0, replicate_physics=True)
    baseline_extension_penalty_weight = 2.0
    best_effort_tilt_weight = 12.0
    motion_curriculum_iterations = 300
    training_progress_steps_per_iteration = 24


@configclass
class DeformableFittedBalancedEnvCfg(DeformableFittedAdaptiveEnvCfg):
    """Retain learned grade correction while penalizing contact loss/chatter."""

    best_effort_contact_gating = False
    baseline_extension_penalty_weight = 1.0
    rewards = OrderedDict(DeformableFittedAdaptiveEnvCfg().rewards)
    rewards["all_wheel_contact"] = 24.0
    rewards["action_rate"] = -0.5
    rewards["action_rate2"] = -0.05


@configclass
class DeformableFittedMobilityEnvCfg(DeformableFittedBalancedEnvCfg):
    """Choose leveling postures that retain measured command tracking."""

    drive_velocity_tracking_weight = 4.0
    drive_yaw_tracking_weight = 0.5
    baseline_extension_penalty_weight = 2.0
    rewards = OrderedDict(DeformableFittedBalancedEnvCfg().rewards)
    rewards["action_rate"] = -2.0


@configclass
class DeformableFittedSafeMobilityEnvCfg(DeformableFittedMobilityEnvCfg):
    """Reserve vertical chassis clearance while rotating on a steep slope."""

    clearance_margin_m = 0.012
    clearance_margin_weight = 40.0
    motion_curriculum_iterations = 100
    rewards = OrderedDict(DeformableFittedMobilityEnvCfg().rewards)
    rewards["all_wheel_contact"] = 40.0
    rewards["wheel_load_balance"] = 4.0


@configclass
class DeformableFittedNativePrecisionEnvCfg(DeformableFittedSafeMobilityEnvCfg):
    """Match native body commands and prioritize actual chassis motion."""

    commands_world_frame = False
    drive_velocity_tracking_weight = 12.0


@configclass
class DeformableFittedSteepExposureEnvCfg(DeformableFittedSafeMobilityEnvCfg):
    """Practice sustained large grades and parking with native play commands."""

    terrain = _make_periodic_slope_terrain(angle_range=(17.0, 20.0), seed=13)
    terrain.terrain_generator.sub_terrains["periodic_slope"].segment_length = 6.0
    commands_world_frame = False
    cmd_lin_vel_x_range = (-0.8, 0.8)
    cmd_lin_vel_y_range = (-0.5, 0.5)
    cmd_ang_vel_z_range = (-1.5, 1.5)
    cmd_rel_standing_envs = 0.35
    cmd_resample_time_range = (3.0, 6.0)
    drive_velocity_tracking_weight = 8.0


@configclass
class DeformableFittedTractionReserveEnvCfg(DeformableFittedSteepExposureEnvCfg):
    """Guide load transfer before the ideal uphill drive margin is exhausted."""

    static_traction_margin_weight = 40.0


@configclass
class DeformableFittedLowProfileEnvCfg(DeformableFittedTractionReserveEnvCfg):
    """Restore the low-body preference without imposing a height termination."""

    soft_body_height_penalty = True
    baseline_extension_penalty_weight = 8.0


@configclass
class DeformableFittedTerrainLowProfileEnvCfg(DeformableFittedTractionReserveEnvCfg):
    """Strengthen the flat low-body preference while retaining steep load transfer."""

    soft_body_height_penalty = True
    height_penalty_weight = 4.0
    flat_baseline_extension_penalty_weight = 14.0


@configclass
class DeformableFittedMixedCornerEnvCfg(DeformableFittedTerrainLowProfileEnvCfg):
    """Train the gentle grades where an indiscriminately low body scraped ground."""

    terrain = _make_periodic_slope_terrain(angle_range=(5.0, 20.0), seed=13)
    terrain.terrain_generator.sub_terrains["periodic_slope"].segment_length = 6.0
    terrain.terrain_generator.sub_terrains["periodic_slope"].angle_choices = (5.0, 10.0, 17.0, 20.0, 20.0)


@configclass
class DeformableFittedKeyboardPlayEnvCfg(_DeformableFittedReal2SimFields, DeformableDynamicKeyboardPlayEnvCfg):
    best_effort_leveling = True
    best_effort_tilt_weight = 6.0
    termination_roll_deg = 45.0
    termination_pitch_deg = 45.0
    real2sim_randomize = False
    real2sim_torque_scale_range = (1., 1.)
    real2sim_command_delay_steps_range = (0, 0)
    real2sim_feedback_delay_steps_range = (1, 1)


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
