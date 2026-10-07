"""Wheel-driven active suspension with encoder odometry and policy-step history."""

from __future__ import annotations

import torch
import trimesh

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from agent_world import AssetPath
from . import cfg_utils as du
from .env import DeformableSuspensionEnv
from .adrc import LegADRC
from .real2sim import Real2SimActuator, load_real2sim_model
from .wheel_drive import WheelVelocityPI
from .traction_guidance import uphill_drive_margin


class DeformableDynamicEnv(DeformableSuspensionEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.enable_chassis_servo or cfg.events is not None:
            raise ValueError("V1 tire dynamics cannot be combined with base servo or wheel material events")
        if cfg.observation_space != 32 * cfg.policy_history_length or cfg.state_space != 40:
            raise ValueError("V1 contract requires policy=32*history_length and critic=40")
        if getattr(cfg, "auto_expand_periodic_terrain", False):
            gen = getattr(cfg.terrain, "terrain_generator", None)
            if gen is not None and "periodic_slope" in gen.sub_terrains:
                old_size = gen.size
                gen.size = du.suspension_training_terrain_size(cfg.scene.num_envs, cfg.scene.env_spacing, old_size)
                if gen.size != old_size:
                    print(f"[INFO]: Expanded shared periodic terrain {old_size} -> {gen.size} "
                          f"for {cfg.scene.num_envs} environments")
        super().__init__(cfg, render_mode, **kwargs)
        if abs(self.physics_dt - cfg.adrc_dt) > 1.0e-9:
            raise ValueError("ADRC must run once per 1 ms physics step; do not subcycle stale measurements")
        self._leg_adrc = LegADRC((self.num_envs, 4), self.device, cfg)
        self._leg_actuator = None
        if getattr(cfg, "real2sim_enabled", False):
            self._leg_actuator = Real2SimActuator(
                (self.num_envs, 4), self.device, cfg, model=load_real2sim_model(cfg.real2sim_model_path)
            )
            mechanism = self._leg_actuator.mechanism
            if mechanism is not None:
                if abs(mechanism.physical_angle_zero-cfg.leg_physical_angle_zero)>1.e-7:
                    raise ValueError("fitted actuator and task physical-angle zeros differ")
                self.robot.write_joint_armature_to_sim(mechanism.armature.expand(self.num_envs,-1),
                                                      joint_ids=self._legs_idx)
        joints = {name: i for i, name in enumerate(self.robot.joint_names)}
        bodies = {name: i for i, name in enumerate(self.robot.body_names)}
        contacts = {name: i for i, name in enumerate(self.contact_sensor.body_names)}
        self._wheels_idx = [joints[name] for name in du.ORDERED_WHEEL_JOINT_NAMES]
        self._wheel_body_ids = [bodies[f"wheel_{i}"] for i in range(1, 5)]
        self._wheel_set_body_ids = [bodies[f"wheel_set_{i}"] for i in range(1, 5)]
        self._wheels_contact_idx = [contacts[f"wheel_{i}"] for i in range(1, 5)]
        self._friction = torch.ones(self.num_envs, 1, device=self.device)
        self._encoder_bias = torch.zeros(self.num_envs, 4, device=self.device)
        self._gyro_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sensor_fifo = torch.zeros(self.num_envs, cfg.max_sensor_delay_steps + 1, 32, device=self.device)
        self._history = torch.zeros(self.num_envs, cfg.policy_history_length, 32, device=self.device)
        self._history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._obs_tick = -1
        self._obs_cache = None
        self._last_wheel_slip = torch.zeros(self.num_envs, 4, device=self.device)
        self._tire_deflection = torch.zeros_like(self._last_wheel_slip)
        self._wheel_drive = WheelVelocityPI((self.num_envs, 4), self.device, cfg)
        self._wheel_target = self._wheel_drive.target
        self._wheel_tau = self._wheel_drive.torque
        self._drive_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._metric_names = ("all_contact", "tilt_square", "height_violation", "clearance_min",
                              "speed_error", "yaw_error", "slip", "torque_saturation",
                              "contact_and_horizontal", "min_wheel_force", "q_limit_fraction",
                              "target_limit_fraction", "target_tracking_error", "residual_tilt_deg",
                              "baseline_extension_rad", "ground_grade_deg", "steep_grade_fraction",
                              "standing_command_fraction", "traction_deficit", "uphill_drive_capacity_n",
                              "uphill_gravity_n")
        self._support_metrics_enabled = (getattr(cfg, "support_gap_weight", 0.0) > 0
                                         or getattr(cfg, "support_load_weight", 0.0) > 0)
        if self._support_metrics_enabled:
            self._metric_names += ("wheel_gap_max_m", "support_gap_cost", "support_load_cost")
        self._metrics = torch.zeros(self.num_envs, len(self._metric_names), device=self.device)
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)

        # Use the actual transformed mesh envelope, not only its base-frame z offsets.
        mesh = trimesh.load(f"{AssetPath}/usd_files/deformable_V2/meshes/base_link.STL", force="mesh")
        vertices = torch.as_tensor(mesh.vertices.copy(), dtype=torch.float32, device=self.device)
        vertices = torch.stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]), dim=-1)
        lo, hi = vertices.amin(0), vertices.amax(0)
        xy = torch.cartesian_prod(torch.linspace(lo[0], hi[0], 7, device=self.device),
                                  torch.linspace(lo[1], hi[1], 7, device=self.device))
        self._bottom_samples = torch.cat((xy, lo[2].expand(len(xy), 1)), dim=-1)
        self._top_vertices = torch.as_tensor(
            trimesh.convex.convex_hull(vertices.cpu().numpy()).vertices.copy(),
            dtype=torch.float32, device=self.device)
        self._set_wheel_material()

    def _set_wheel_material(self):
        # PhysX handles normal collision; our tire law owns ALL wheel tangential friction.
        view = self.robot.root_physx_view
        materials = view.get_material_properties()
        start = 0
        for body_id, path in enumerate(view.link_paths[0]):
            count = self.robot._physics_sim_view.create_rigid_body_view(path).max_shapes
            if body_id in self._wheel_body_ids:
                materials[:, start:start + count, :2] = 0.0
                materials[:, start:start + count, 2] = 0.0
            start += count
        if start != view.max_shapes:
            raise RuntimeError("Wheel material shape mapping mismatch")
        view.set_material_properties(materials, torch.arange(self.num_envs, dtype=torch.int32, device="cpu"))

    def _ground_height(self, points):
        if self._periodic:
            return du.periodic_slope_height_torch(
                points[..., 0] + self._profile_x_offset, self._period_seg, self._slope_angle_table)
        shape = (self.num_envs,) + (1,) * (points.ndim - 2)
        return self.scene.env_origins[:, 2].reshape(shape).expand(points.shape[:-1])

    def _normals_at(self, points):
        normals = torch.zeros_like(points)
        normals[..., 2] = 1.0
        if self._periodic:
            x = points[..., 0] + self._profile_x_offset
            eps = 0.01
            gradient = (du.periodic_slope_height_torch(x + eps, self._period_seg, self._slope_angle_table)
                        - du.periodic_slope_height_torch(x - eps, self._period_seg, self._slope_angle_table)) / (2 * eps)
            normals[..., 0] = -gradient
        return torch.nn.functional.normalize(normals, dim=-1)

    def _wheel_geometry_w(self):
        data = self.robot.data
        quats = data.body_link_quat_w[:, self._wheel_body_ids]
        offset = torch.zeros(self.num_envs, 4, 3, device=self.device)
        offset[..., 0] = -0.0055
        centers = data.body_link_pos_w[:, self._wheel_body_ids] + quat_apply(quats, offset)
        normals = self._normals_at(centers)
        axle = torch.zeros_like(centers)
        axle[..., 0] = 1.0
        axle = quat_apply(quats, axle)
        roll = torch.nn.functional.normalize(torch.cross(axle, normals, dim=-1), dim=-1)
        points = centers - du.WHEEL_RADIUS * normals
        return centers, points, normals, axle, roll

    @property
    def wheel_normal_forces(self):
        _, _, normals, _, _ = self._wheel_geometry_w()
        forces = self.contact_sensor.data.net_forces_w[:, self._wheels_contact_idx]
        return (forces * normals).sum(-1).clamp_min(0.0)

    @property
    def chassis_clearance(self):
        data = self.robot.data
        quat = data.root_link_quat_w[:, None].expand(-1, len(self._bottom_samples), -1)
        points = data.root_link_pos_w[:, None] + quat_apply(
            quat, self._bottom_samples[None].expand(self.num_envs, -1, -1))
        return (points[..., 2] - self._ground_height(points)).amin(-1)

    @property
    def body_top_height(self):
        data = self.robot.data
        quat = data.root_link_quat_w[:, None].expand(-1, len(self._top_vertices), -1)
        top_z = (data.root_link_pos_w[:, None] + quat_apply(
            quat, self._top_vertices[None].expand(self.num_envs, -1, -1)))[..., 2].amax(-1)
        _, points, _, _, _ = self._wheel_geometry_w()
        # Ground surface at each wheel, so an airborne wheel cannot hide excess height.
        return top_z - self._ground_height(points).amax(-1)

    def _drive_cmd_b(self, command=None):
        cmd = (self.cmd_buf if command is None else command).clone()
        if self.cfg.commands_world_frame:
            horizontal = torch.cat((cmd[:, :2], torch.zeros_like(cmd[:, :1])), dim=-1)
            cmd[:, :2] = quat_apply_inverse(self.robot.data.root_link_quat_w, horizontal)[:, :2]
        return cmd

    def _filter_drive_command(self):
        delta = self.cmd_buf - self._drive_command
        scale = (self.cfg.drive_linear_acceleration_limit * self.physics_dt
                 / delta[:, :2].norm(dim=-1, keepdim=True).clamp_min(1.0e-9)).clamp(max=1.0)
        self._drive_command[:, :2].add_(delta[:, :2] * scale)
        yaw_step = self.cfg.drive_yaw_acceleration_limit * self.physics_dt
        self._drive_command[:, 2].add_(delta[:, 2].clamp(-yaw_step, yaw_step))

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if self.cfg.external_cmd_override:
            return
        progress = min(1.0, self.common_step_counter / max(
            1, self.cfg.motion_curriculum_iterations * self.cfg.training_progress_steps_per_iteration))
        # Standing -> translation -> spin -> simultaneous motion; both spin signs are sampled.
        linear_scale = min(1.0, max(0.0, (progress - 0.05) / 0.35))
        yaw_scale = min(1.0, max(0.0, (progress - 0.25) / 0.65))
        if getattr(self.cfg, "support_motion_commands", False):
            self.cmd_buf[env_ids] = du.suspension_motion_commands(
                torch.rand(len(env_ids), 7, device=self.device),
                (self.cfg.cmd_lin_vel_x_range[1], self.cfg.cmd_lin_vel_y_range[1],
                 self.cfg.cmd_ang_vel_z_range[1]),
                linear_scale, yaw_scale, self.cfg.cmd_rel_standing_envs)
            return
        self.cmd_buf[env_ids, :2] *= linear_scale
        self.cmd_buf[env_ids, 2] *= yaw_scale
        mode = torch.randint(0, 5, (len(env_ids),), device=self.device)
        self.cmd_buf[env_ids[mode == 0]] = 0.0
        self.cmd_buf[env_ids[mode == 1], 2] = 0.0
        self.cmd_buf[env_ids[mode == 2], :2] = 0.0
        full_spin = env_ids[mode == 3]
        self.cmd_buf[full_spin, 2] = torch.sign(self.cmd_buf[full_spin, 2]) * yaw_scale * self.cfg.cmd_ang_vel_z_range[1]

    def _pre_physics_step(self, actions):
        previous_target = self.leg_target.clone()
        super()._pre_physics_step(actions.clamp(-1.0, 1.0))
        if self.cfg.action_contract_version == "legacy_v1":
            self.leg_target = (self.q_cmd[:, None] - du.Q_LOW * (-self.actions).clamp_min(0.0)).clamp(
                du.LEG_LOWER_LIMIT, self.cfg.leg_target_upper_limit)
        else:
            self.leg_target = du.suspension_target(self.actions, self.q_cmd[:, None], self.cfg.leg_target_upper_limit,
                                                  getattr(self.cfg,"leg_target_lower_limit",du.LEG_LOWER_LIMIT))
            max_change = self.cfg.leg_target_rate_limit * self.step_dt
            self.leg_target = previous_target + (self.leg_target - previous_target).clamp(-max_change, max_change)

    def _apply_action(self):
        data = self.robot.data
        q = data.joint_pos[:, self._legs_idx]
        qd = data.joint_vel[:, self._legs_idx]
        if self._leg_actuator is not None:
            measured_q, _, _ = self._leg_actuator.sensor_measurement(q, qd)
            raw_current = self._leg_adrc.update(
                measured_q, self.leg_target,
                applied_current_raw=self._leg_actuator.command_current_raw,
            )
            leg_tau = self._leg_actuator.apply(raw_current, q, qd)
        else:
            leg_tau = self._leg_adrc.update(q, self.leg_target)
        self.robot.set_joint_effort_target(leg_tau, joint_ids=self._legs_idx)
        _, points, normals, _, roll = self._wheel_geometry_w()
        data = self.robot.data
        q = data.joint_pos[:, self._legs_idx]
        self._filter_drive_command()
        target = (du.omni_matrix(q) @ self._drive_cmd_b(self._drive_command).unsqueeze(-1)).squeeze(-1)
        wheel_speed = data.joint_vel[:, self._wheels_idx]
        load = self.wheel_normal_forces
        # A stale force sample must never keep an already-separated tire active.
        gap = points[..., 2] - self._ground_height(points)
        load = torch.where(gap <= self.cfg.tire_contact_gap, load, torch.zeros_like(load))
        self._wheel_drive.update(target, wheel_speed, load >= self.cfg.wheel_contact_force_threshold)
        self.robot.set_joint_effort_target(self._wheel_tau, joint_ids=self._wheels_idx)

        # Wheel-set motion includes suspension rates, but excludes driven wheel spin.
        ws_ids = self._wheel_set_body_ids
        velocity = data.body_com_lin_vel_w[:, ws_ids] + torch.cross(
            data.body_com_ang_vel_w[:, ws_ids],
            points - data.body_com_pos_w[:, ws_ids], dim=-1)
        lateral = torch.nn.functional.normalize(torch.cross(normals, roll, dim=-1), dim=-1)
        self._last_wheel_slip.copy_(du.WHEEL_RADIUS * wheel_speed - (velocity * roll).sum(-1))
        # A compliant tread patch can carry force at zero slip speed. Bound its
        # deflection by the available friction so sliding cannot wind up stored force;
        # zero load clears the patch, including during separation/landing.
        deflection_limit = self._friction * load / self.cfg.tire_contact_stiffness
        self._tire_deflection.copy_((self._tire_deflection + self.physics_dt * self._last_wheel_slip).clamp(
            -deflection_limit, deflection_limit))
        fx, fy = du.wheel_traction(self._last_wheel_slip, (velocity * lateral).sum(-1),
                                  load, self._friction, self.cfg.tire_slip_stiffness,
                                  self.cfg.tire_lateral_drag,
                                  elastic_force=self.cfg.tire_contact_stiffness * self._tire_deflection)
        forces = fx[..., None] * roll + fy[..., None] * lateral
        # Force at contact generates wheel reaction torque r x F; motor effort supplies
        # equal/opposite axle reactions through the articulation, without duplicate torque.
        wheel_quat = data.body_link_quat_w[:, self._wheel_body_ids]
        torque_w = torch.cross(points - data.body_com_pos_w[:, self._wheel_body_ids], forces, dim=-1)
        # Explicit COM wrench avoids the permanent composer's cached global link poses
        # in Isaac Lab 2.3.2. PhysX applies these local forces at the center of mass.
        self.robot.set_external_force_and_torque(
            quat_apply_inverse(wheel_quat, forces), quat_apply_inverse(wheel_quat, torque_w),
            body_ids=self._wheel_body_ids, is_global=False)
        self._last_servo_force.copy_(forces.sum(1))
        self._last_servo_torque_z.copy_(torch.cross(points - data.root_com_pos_w[:, None], forces, dim=-1)[..., 2].sum(1))

    def _get_observations(self):
        if self._obs_tick == self.common_step_counter and self._obs_cache is not None:
            return self._obs_cache
        data = self.robot.data
        q = data.joint_pos[:, self._legs_idx]
        if self._leg_actuator is not None:
            q_obs, qd_obs, current_obs = self._leg_actuator.sensor_state()
            effort_obs = current_obs / self._leg_actuator.current_limit
        else:
            q_obs = q
            qd_obs = data.joint_vel[:, self._legs_idx]
            effort_obs = data.applied_torque[:, self._legs_idx] * 0.05
        wheel_speed = data.joint_vel[:, self._wheels_idx] + self._encoder_bias
        wheel_speed = wheel_speed + torch.randn_like(wheel_speed) * self.cfg.encoder_noise_std
        twist = du.estimate_twist(q_obs, wheel_speed)
        gyro = data.root_ang_vel_b + self._gyro_bias + torch.randn_like(data.root_ang_vel_b) * self.cfg.gyro_noise_std
        gravity = data.projected_gravity_b + torch.randn_like(data.projected_gravity_b) * self.cfg.gravity_noise_std
        # 32 = old 26 + wheel encoder4 + encoder-derived vx/vy2. No true velocity leaks.
        frame = torch.cat((self.q_cmd[:, None], self._drive_cmd_b() * q.new_tensor((1.0, 1.0, 0.25)),
                           gyro * 0.5, gravity, q_obs, qd_obs * 0.1,
                           effort_obs, self.actions,
                           wheel_speed * 0.05, twist[:, :2]), dim=-1)
        frame = torch.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)
        fresh = ~self._history_valid
        self._sensor_fifo = torch.roll(self._sensor_fifo, -1, dims=1)
        self._sensor_fifo[:, -1] = frame
        self._sensor_fifo[fresh] = frame[fresh, None]
        delayed = self._sensor_fifo[torch.arange(self.num_envs, device=self.device),
                                    self.cfg.max_sensor_delay_steps - self._delay].clone()
        # Command and prior action are local controller state, not delayed sensors.
        delayed[:, :4] = frame[:, :4]
        delayed[:, 22:26] = frame[:, 22:26]
        self._history = torch.roll(self._history, -1, dims=1)
        self._history[:, -1] = delayed
        self._history[fresh] = delayed[fresh, None]
        self._history_valid[:] = True
        legacy = self.cfg.action_contract_version == "legacy_v1"
        critic = torch.cat((frame, data.root_lin_vel_b, self.body_top_height[:, None] * (1.0 if legacy else 5.0),
                             self.wheel_normal_forces * (1.0 if legacy else 0.02)), dim=-1)
        if not legacy:
            critic = torch.nan_to_num(critic, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)
        self._obs_cache = {"policy": self._history.flatten(1).clone(), "critic": critic.clone()}
        if getattr(self.cfg, "training_steep_reference_mix", False):
            _, _, normals, _, _ = self._wheel_geometry_w()
            mix = du.suspension_steep_reference_mix(
                normals, self.cfg.steep_teacher_start_grade_deg, self.cfg.steep_teacher_full_grade_deg)
            # Rollout storage retains this privileged training selector, but
            # neither policy nor critic observation group includes it.
            self._obs_cache["training_steep_reference_mix"] = torch.nan_to_num(mix)[:, None].detach()
        self._obs_tick = self.common_step_counter
        return self._obs_cache

    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        height_failure = (self.body_top_height > self.cfg.max_body_top_height + self.cfg.height_termination_margin)
        if self.cfg.enforce_tunnel_height:
            terminated |= height_failure & (self.episode_length_buf > self.cfg.height_settle_steps)
        return terminated, timeout

    def _get_rewards(self):
        reward = super()._get_rewards()
        h = self.body_top_height
        q = self.robot.data.joint_pos[:, self._legs_idx]
        clearance = self.chassis_clearance
        # Anchor the lowest corner while allowing unequal leg angles to level
        # the body on a slope. Reward changes cannot change a frozen play policy.
        extension = du.suspension_baseline_extension(q, self.q_cmd)
        reward += self.cfg.baseline_reward_weight * torch.exp(-extension.square() / 0.0025)
        reward -= getattr(self.cfg, "baseline_extension_penalty_weight", 0.0) * extension
        clearance_weight = getattr(self.cfg, "clearance_margin_weight", 0.0)
        if clearance_weight:
            grade_margin = getattr(self.cfg, "clearance_grade_margin_m", 0.0)
            if grade_margin > 0:
                normal = self._normals_at(self.robot.data.root_link_pos_w[:, None])[:, 0]
                clearance_cost = du.suspension_grade_clearance_cost(
                    clearance, normal, self.cfg.clearance_margin_m, grade_margin,
                    motion_command=self._drive_command,
                    steep_motion_margin_m=getattr(self.cfg, "clearance_steep_motion_margin_m", 0.),
                    attitude_rate=self.robot.data.root_ang_vel_b[:, :2])
            else:
                clearance_cost = du.suspension_clearance_cost(clearance, self.cfg.clearance_margin_m)
            reward -= clearance_weight * clearance_cost.clamp(max=25.0)
        linear_weight = getattr(self.cfg, "drive_velocity_tracking_weight", 0.0)
        yaw_weight = getattr(self.cfg, "drive_yaw_tracking_weight", 0.0)
        if linear_weight or yaw_weight:
            data = self.robot.data
            velocity = data.root_link_lin_vel_w[:, :2] if self.cfg.commands_world_frame else data.root_link_lin_vel_b[:, :2]
            linear_cost, yaw_cost = du.suspension_drive_tracking_cost(
                velocity, data.root_ang_vel_b[:, 2], self._drive_command)
            # The drive command already includes the configured acceleration
            # limits; unavoidable command jumps must not dominate learning.
            reward -= linear_weight * linear_cost.clamp(max=25.0)
            reward -= yaw_weight * yaw_cost.clamp(max=25.0)
        data = self.robot.data
        ground_normal = self._normals_at(data.root_link_pos_w[:, None])[:, 0]
        flat_extension_weight = getattr(self.cfg, "flat_baseline_extension_penalty_weight", 0.0)
        if flat_extension_weight:
            reward -= flat_extension_weight * extension * du.suspension_flat_grade_gate(
                ground_normal, self.cfg.low_profile_fade_grade_deg)
        spread_weight = getattr(self.cfg, "flat_leg_spread_weight", 0.0)
        if spread_weight:
            _, _, wheel_normals, _, _ = self._wheel_geometry_w()
            reward -= spread_weight * du.suspension_flat_leg_spread_cost(q, wheel_normals).clamp(max=25.)
        traction_cost = drive_capacity = slope_gravity = self._friction[:, 0] * 0.0
        traction_weight = getattr(self.cfg, "static_traction_margin_weight", 0.0)
        if traction_weight:
            _, _, _, _, roll = self._wheel_geometry_w()
            traction_cost, drive_capacity, slope_gravity = uphill_drive_margin(
                self.wheel_normal_forces, roll, ground_normal, self._friction,
                mass_kg=self.cfg.traction_mass_kg,
                wheel_torque_limit_nm=self.cfg.wheel_torque_limit,
                wheel_radius_m=du.WHEEL_RADIUS,
                reserve_fraction=self.cfg.traction_reserve_fraction,
            )
            reward -= traction_weight * traction_cost
        if self.cfg.enforce_tunnel_height or getattr(self.cfg, "soft_body_height_penalty", False):
            height_excess = du.suspension_body_height_cost(
                h, self.cfg.max_body_top_height, ground_normal,
                flat_fade_grade_deg=(self.cfg.low_profile_fade_grade_deg
                                    if getattr(self.cfg, "height_penalty_flat_only", False) else None),
            )
            reward -= self.cfg.height_penalty_weight * height_excess
        if getattr(self.cfg, "best_effort_leveling", False):
            contact_ratio = (self.wheel_normal_forces.amin(-1)
                             / self.cfg.wheel_contact_force_threshold).clamp(0.0, 1.0)
            precision_multiplier = du.suspension_grade_precision_multiplier(
                ground_normal, getattr(self.cfg, "gentle_precision_multiplier", 1.))
            reward -= self.cfg.best_effort_tilt_weight * precision_multiplier * du.suspension_tilt_cost(
                self.robot.data.projected_gravity_b, contact_ratio,
                gate_by_contact=getattr(self.cfg, "best_effort_contact_gating", True))
        if getattr(self, "_support_metrics_enabled", False):
            _, wheel_points, _, _, _ = self._wheel_geometry_w()
            wheel_gap = wheel_points[..., 2] - self._ground_height(wheel_points)
            gap_cost, load_cost = du.suspension_support_costs(
                wheel_gap, self.wheel_normal_forces,
                gap_tolerance_m=self.cfg.support_gap_tolerance_m,
                gap_scale_m=self.cfg.support_gap_scale_m, min_load_n=self.cfg.support_min_load_n)
            reward -= self.cfg.support_gap_weight * gap_cost + self.cfg.support_load_weight * load_cost
        cmd = self._drive_cmd_b()
        if self._leg_actuator is not None:
            saturated = (self._leg_actuator.command_current_raw.abs() >= .99*self._leg_actuator.current_limit)
        else:
            saturated = self.robot.data.applied_torque[:,self._legs_idx].abs() >= .99*self.cfg.max_leg_torque
        lower_limit = getattr(self.cfg,"leg_target_lower_limit",du.LEG_LOWER_LIMIT)
        data = self.robot.data
        measured_velocity = data.root_link_lin_vel_w if self.cfg.commands_world_frame else data.root_link_lin_vel_b
        ground_grade = torch.atan2(ground_normal[:, :2].norm(dim=-1), ground_normal[:, 2]) * (180.0 / torch.pi)
        metrics = torch.stack((
            (self.wheel_normal_forces > self.cfg.wheel_contact_force_threshold).all(-1).float(),
            self.robot.data.projected_gravity_b[:, :2].square().sum(-1),
            (h > self.cfg.max_body_top_height).float(), clearance,
            (measured_velocity[:, :2] - self._drive_command[:, :2]).norm(dim=-1),
            (self.robot.data.root_ang_vel_b[:, 2] - cmd[:, 2]).abs(),
            self._last_wheel_slip.abs().mean(-1),
            saturated.float().mean(-1),
            ((self.wheel_normal_forces > self.cfg.wheel_contact_force_threshold).all(-1)
             & (self.robot.data.projected_gravity_b[:, :2].norm(dim=-1)
                < torch.sin(q.new_tensor(self.cfg.horizontal_tolerance_deg * torch.pi / 180.0)))).float(),
            self.wheel_normal_forces.amin(-1),
            ((q < lower_limit + 0.02)
             | (q > self.cfg.leg_target_upper_limit - 0.02)).float().mean(-1),
            ((self.leg_target < lower_limit + 0.02)
             | (self.leg_target > self.cfg.leg_target_upper_limit - 0.02)).float().mean(-1),
            (q - self.leg_target).abs().mean(-1),
            torch.atan2(self.robot.data.projected_gravity_b[:, :2].norm(dim=-1),
                        -self.robot.data.projected_gravity_b[:, 2]) * (180.0 / torch.pi),
            extension,
            ground_grade,
            (ground_grade >= 17.0).float(),
            (self._drive_command.abs().amax(-1) < 1.e-6).float(),
            traction_cost,
            drive_capacity,
            slope_gravity,
        ), dim=-1)
        if getattr(self, "_support_metrics_enabled", False):
            metrics = torch.cat((metrics, torch.stack(
                (wheel_gap.clamp_min(0.0).amax(-1), gap_cost, load_cost), dim=-1)), dim=-1)
        settled = (self.episode_length_buf > self.cfg.height_settle_steps).float()
        if getattr(self.cfg, "auto_expand_periodic_terrain", False):
            # Episode metrics above exclude short failed episodes. Show their
            # population explicitly so surviving fragments cannot look healthy.
            # The runner retains log dictionaries across the rollout. Give it
            # a fresh snapshot rather than overwriting earlier step samples.
            log = dict(self.extras["log"])
            log["health/settled_env_fraction"] = settled.mean().item()
            log["health/terminated_env_fraction"] = self.reset_terminated.float().mean().item()
            log["health/timeout_env_fraction"] = self.reset_time_outs.float().mean().item()
            if self._periodic:
                half = q.new_tensor(self.cfg.terrain.terrain_generator.size) / 2
                outside = (data.root_link_pos_w[:, :2].abs() > half - 0.8).any(-1)
                log["health/terrain_outside_fraction"] = outside.float().mean().item()
            self.extras["log"] = log
        self._metrics += metrics * settled[:, None]
        self._metric_steps += settled
        return self.cfg.reward_scale * reward.clamp(-self.cfg.reward_total_clip, self.cfg.reward_total_clip)

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        valid_metrics = self._metric_steps[env_ids] > 0
        means = self._metrics[env_ids] / self._metric_steps[env_ids, None].clamp_min(1.0)
        super()._reset_idx(env_ids)
        if self._periodic:
            data = self.robot.data
            pose = torch.cat((data.root_link_pos_w[env_ids], data.root_link_quat_w[env_ids]), dim=-1).clone()

            def contact_height(q):
                centers, _ = du.wheel_geometry(q)
                shape = (len(env_ids),) + (1,) * (q.ndim - 1)
                rotated = quat_apply(pose[:, 3:].reshape(*shape, 4).expand(*q.shape, 4), centers)
                x = pose[:, 0].reshape(shape) + rotated[..., 0] + self._profile_x_offset
                ground = du.periodic_slope_height_torch(x, self._period_seg, self._slope_angle_table)
                k = torch.floor(x / (4.0 * self._period_seg)).long().clamp(0, self._slope_angle_table.numel() - 1)
                phase = x - k.to(x.dtype) * (4.0 * self._period_seg)
                slope = torch.tan(self._slope_angle_table[k] * (torch.pi / 180.0))
                slope = torch.where(((phase >= 0) & (phase < self._period_seg))
                                    | ((phase >= 2 * self._period_seg) & (phase < 3 * self._period_seg)),
                                    slope, torch.zeros_like(slope))
                nz = torch.rsqrt(1.0 + slope.square())
                return ground - rotated[..., 2] + du.WHEEL_RADIUS / nz, nz

            bottom = quat_apply(pose[:, None, 3:].expand(-1, len(self._bottom_samples), -1),
                                self._bottom_samples[None].expand(len(env_ids), -1, -1))
            bottom_ground = du.periodic_slope_height_torch(
                pose[:, None, 0] + bottom[..., 0] + self._profile_x_offset,
                self._period_seg, self._slope_angle_table)
            clearance_height = (bottom_ground - bottom[..., 2]).amax(-1) + self.cfg.chassis_ground_threshold + 1.e-6
            samples = torch.linspace(getattr(self.cfg,"leg_target_lower_limit",du.LEG_LOWER_LIMIT),
                                     self.cfg.leg_target_upper_limit, 65,
                                     device=self.device, dtype=pose.dtype)
            sampled_q = samples[None, :, None].expand(len(env_ids), -1, 4)
            sampled_height, _ = contact_height(sampled_q)
            low_height = sampled_height.amin(1).amax(-1).maximum(clearance_height)
            high_height = sampled_height.amax(1).amin(-1)
            baseline, _ = contact_height(self.q_cmd[env_ids, None].expand(-1, 4))
            height = baseline.amax(-1).maximum(low_height).minimum(high_height)
            # Sample brackets also expose the small jumps in radius/n_z at slope breaks.
            crossing = (sampled_height[:, :-1] >= height[:, None, None]) & (sampled_height[:, 1:] <= height[:, None, None])
            bracket = crossing.to(torch.int64).argmax(1)
            lower, upper = samples[bracket], samples[bracket + 1]
            for _ in range(24):
                mid = 0.5 * (lower + upper)
                mid_height, _ = contact_height(mid)
                too_high = mid_height > height[:, None]
                lower = torch.where(too_high, mid, lower)
                upper = torch.where(too_high, upper, mid)
            q = 0.5 * (lower + upper)
            solved_height, _ = contact_height(q)
            feasible = ((low_height <= high_height) & crossing.any(1).all(-1)
                        & ((solved_height - height[:, None]).abs().amax(-1) <= 2.e-6))
            # Infeasible resets may be airborne, but must not embed wheels or chassis.
            safe_height = solved_height.amax(-1).maximum(clearance_height)
            height = torch.where(feasible, height, safe_height)
            pose[:, 2] = height.maximum(safe_height) + max(0.0, self.cfg.reset_height_buffer) + 1.e-6
            if getattr(self.cfg, "best_effort_leveling", False) and (~feasible).any():
                # A horizontal reset is impossible here. Start tangent to the
                # local plane instead, then let the policy reduce residual tilt.
                ids = (~feasible).nonzero(as_tuple=False).squeeze(-1)
                x = pose[ids, 0] + self._profile_x_offset
                eps = 0.01
                slope = (du.periodic_slope_height_torch(x + eps, self._period_seg, self._slope_angle_table)
                         - du.periodic_slope_height_torch(x - eps, self._period_seg, self._slope_angle_table)) / (2 * eps)
                theta = torch.atan(slope)
                yaw = 2 * torch.atan2(pose[ids, 6], pose[ids, 3])
                ct, st = torch.cos(theta / 2), torch.sin(theta / 2)
                cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
                pose[ids, 3:] = torch.stack((ct * cy, -st * sy, -st * cy, ct * sy), -1)
                q[ids] = self.q_cmd[env_ids[ids], None]
                centers, _ = du.wheel_geometry(q[ids])
                rotated = quat_apply(pose[ids, None, 3:].expand(-1, 4, -1), centers)
                wx = pose[ids, None, 0] + rotated[..., 0] + self._profile_x_offset
                ground = du.periodic_slope_height_torch(wx, self._period_seg, self._slope_angle_table)
                wheel_height = ground - rotated[..., 2] + du.WHEEL_RADIUS / torch.cos(theta[:, None])
                bottom[ids] = quat_apply(pose[ids, None, 3:].expand(-1, len(self._bottom_samples), -1),
                                         self._bottom_samples[None].expand(len(ids), -1, -1))
                bottom_ground[ids] = du.periodic_slope_height_torch(
                    pose[ids, None, 0] + bottom[ids, :, 0] + self._profile_x_offset,
                    self._period_seg, self._slope_angle_table)
                body_height = (bottom_ground[ids] - bottom[ids, :, 2]).amax(-1) + self.cfg.chassis_ground_threshold
                pose[ids, 2] = wheel_height.amax(-1).maximum(body_height) + max(0.0, self.cfg.reset_height_buffer) + 1.e-6
                solved_height[ids] = wheel_height
            joints = data.joint_pos[env_ids].clone()
            joints[:, self._legs_idx] = q
            # Resolve passive joint order explicitly for unequal per-corner targets.
            name_to_id = {name: i for i, name in enumerate(self.robot.joint_names)}
            joints[:, [name_to_id[n] for n in du.ORDERED_WS_JOINT_NAMES]] = q
            joints[:, [name_to_id[n] for n in du.ORDERED_UPPER_LEG_JOINT_NAMES]] = -q
            self.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
            self.robot.write_root_pose_to_sim(pose, env_ids=env_ids)
            self.leg_target[env_ids] = q
            initial_actions = du.suspension_action(q, self.q_cmd[env_ids, None], self.cfg.leg_target_upper_limit,
                                                   getattr(self.cfg,"leg_target_lower_limit",du.LEG_LOWER_LIMIT))
            self.actions[env_ids] = initial_actions
            self.last_actions[env_ids] = initial_actions
            self._prev2_actions[env_ids] = initial_actions
            gap = pose[:, 2, None] - solved_height
            self.extras["log"]["dynamic/reset_gap_max"] = gap.amax().item()
            self.extras["log"]["dynamic/reset_penetration_max"] = (-gap).clamp_min(0.0).amax().item()
            self.extras["log"]["dynamic/reset_infeasible_fraction"] = (~feasible).float().mean().item()
            self.extras["log"]["dynamic/reset_clearance_min"] = (pose[:, 2, None] + bottom[..., 2] - bottom_ground).amin().item()
        leg_q = self.robot.data.joint_pos[:, self._legs_idx]
        if self._leg_actuator is not None:
            leg_qd = self.robot.data.joint_vel[:, self._legs_idx]
            self._leg_actuator.reset(env_ids, leg_q, leg_qd)
            measured_q, _, _ = self._leg_actuator.sensor_state()
        else:
            measured_q = leg_q
        self._leg_adrc.reset(env_ids, measured_q[env_ids], self.leg_target[env_ids])
        self._wheel_drive.reset(env_ids)
        self._drive_command[env_ids] = 0.0
        self._last_wheel_slip[env_ids] = 0.0
        self._tire_deflection[env_ids] = 0.0
        for i, name in enumerate(self._metric_names):
            if valid_metrics.any():
                self.extras["log"][f"dynamic/{name}"] = means[valid_metrics, i].mean().item()
        self.extras["log"]["dynamic/valid_episode_fraction"] = valid_metrics.float().mean().item()
        self._metrics[env_ids] = 0.0
        self._metric_steps[env_ids] = 0.0
        self._friction[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self.cfg.tire_friction_range)
        self._encoder_bias[env_ids] = torch.randn(len(env_ids), 4, device=self.device) * self.cfg.encoder_bias_std
        self._gyro_bias[env_ids] = torch.randn(len(env_ids), 3, device=self.device) * self.cfg.gyro_bias_std
        self._delay[env_ids] = torch.randint(0, self.cfg.max_sensor_delay_steps + 1, (len(env_ids),), device=self.device)
        self._history_valid[env_ids] = False
        self._history[env_ids] = 0.0
        self._sensor_fifo[env_ids] = 0.0
        self._obs_tick = -1
        self._obs_cache = None
        zeros = torch.zeros(len(env_ids), 4, 3, device=self.device)
        self.robot.set_external_force_and_torque(zeros, zeros, body_ids=self._wheel_body_ids,
                                                env_ids=env_ids, is_global=False)
