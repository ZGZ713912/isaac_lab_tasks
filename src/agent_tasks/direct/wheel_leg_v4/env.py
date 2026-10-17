from __future__ import annotations

import torch

from agent_tasks.direct.wheel_leg_v3.wheel_leg_task.env import WheelLegV3Env
from .contract import V4_ACTOR_DIM, validate_v4_shapes
from .control import compute_v4_torques


class WheelLegV4Env(WheelLegV3Env):
    """Wheel_leg_V3 asset with V5 explicit control and 230D actor history."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._v4_history = torch.zeros(
            self.num_envs, 5, 46, dtype=torch.float32, device=self.device
        )
        self._v4_history_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._v4_prev_spring_compression = torch.zeros(
            self.num_envs, 2, dtype=torch.float32, device=self.device
        )
        self._v4_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._v4_phase_time = torch.zeros(self.num_envs, device=self.device)
        self.torques = torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
        self._v4_leg_targets = torch.zeros(self.num_envs, 4, device=self.device)
        self._v4_wheel_targets = torch.zeros(self.num_envs, 2, device=self.device)

    def _base_height(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    def _contact_magnitudes(self) -> torch.Tensor:
        history = self.contact_sensor.data.net_forces_w_history
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=1)

    def _wheel_contact_flags(self) -> torch.Tensor:
        contact = self._contact_magnitudes()
        wheel_ids = torch.as_tensor(self._desired_contact_link_idx, device=self.device)
        return (contact[:, wheel_ids] > self.cfg.v4_wheel_contact_threshold).to(torch.float32)

    def _wheel_slip(self) -> torch.Tensor:
        wheel_ids = torch.as_tensor(self._wheel_link_idx, device=self.device)
        lin = self.robot.data.body_lin_vel_w[:, wheel_ids]
        ang = self.robot.data.body_ang_vel_w[:, wheel_ids]
        offset = torch.zeros_like(lin)
        offset[..., 2] = -0.06
        bottom_velocity = lin + torch.cross(ang, offset, dim=-1)
        return bottom_velocity[..., :2].square().sum(dim=-1)

    def _v4_command(self) -> torch.Tensor:
        # V3 stores [vx, vy, yaw] and keeps height in a separate command. V4's
        # contract is [vx, yaw, height], matching the V5 observation contract.
        return torch.stack((self.command[:, 0], self.command[:, 2], self.height_cmd), dim=-1)

    def _v4_policy_state(self):
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        leg_ids = self._legs_act_idx
        wheel_ids = self._wheel_idx
        q = torch.cat((joint_pos[:, leg_ids], joint_pos[:, wheel_ids]), dim=-1)
        dq = torch.cat((joint_vel[:, leg_ids], joint_vel[:, wheel_ids]), dim=-1)
        default = torch.cat(
            (self.robot.data.default_joint_pos[:, leg_ids],
             self.robot.data.default_joint_pos[:, wheel_ids]), dim=-1
        )
        return q, dq, default

    def _v4_spring_observation(self):
        if self._gas_spring_prev_length is None:
            zeros = torch.zeros(self.num_envs, 2, device=self.device)
            return zeros, zeros
        compression = self._gas_spring_nominal_length.unsqueeze(0) - self._gas_spring_prev_length
        rate = (compression - self._v4_prev_spring_compression) / max(self.step_dt, 1.0e-6)
        self._v4_prev_spring_compression.copy_(compression.detach())
        return compression, rate

    def _v4_task_fields(self, contact):
        cmd = self._v4_command()
        moving = cmd[:, 0].abs() > 0.05
        turning = cmd[:, 2].abs() > 0.05
        height_delta = cmd[:, 1] - float(self.cfg.default_height_cmd)
        mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        mode[moving | turning] = 1
        mode[(~moving & ~turning) & (height_delta > 0.015)] = 2
        mode[(~moving & ~turning) & (height_delta < -0.015)] = 3

        grounded = contact.all(dim=-1)
        airborne = ~grounded
        phase = self._v4_phase.clone()
        phase[grounded & (phase == 2)] = 3
        phase[grounded & (phase == 3) & (self._v4_phase_time > 0.1)] = 4
        phase[grounded & (phase == 4) & (self._v4_phase_time > 0.5)] = 0
        phase[airborne & (phase == 0)] = 1
        phase[airborne & (phase == 1)] = 2
        changed = phase != self._v4_phase
        self._v4_phase_time += self.step_dt
        self._v4_phase_time[changed] = 0.0
        self._v4_phase = phase

        mode_onehot = torch.nn.functional.one_hot(mode, 5).to(torch.float32)
        phase_onehot = torch.nn.functional.one_hot(phase, 5).to(torch.float32)
        upper_target = torch.stack(
            (
                torch.zeros_like(cmd[:, 0]),
                height_delta,
                torch.zeros_like(cmd[:, 0]),
                (phase == 1).to(cmd.dtype),
            ), dim=-1,
        )
        return mode_onehot, upper_target, phase_onehot, self._v4_phase_time[:, None]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 6):
            raise ValueError("WheelLegV4 expects N x 6 actions")
        self.last_actions.copy_(self._actions)
        self._previous_actions.copy_(self._actions)
        safe = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        leg_action = safe[:, :4].clamp(-self.cfg.v4_action_clip, self.cfg.v4_action_clip)
        wheel_action = safe[:, 4:6].clamp(
            -self.cfg.v4_wheel_action_clip, self.cfg.v4_wheel_action_clip
        )
        self._actions.copy_(torch.cat((leg_action, wheel_action), dim=-1))
        _, _, default = self._v4_policy_state()
        self._v4_leg_targets = default[:, :4] + leg_action * self.cfg.v4_leg_position_scale
        signs = torch.as_tensor(self.cfg.v4_wheel_joint_sign, device=self.device)
        self._v4_wheel_targets = wheel_action * self.cfg.v4_wheel_velocity_scale * signs

    def _apply_action(self) -> None:
        q, dq, _ = self._v4_policy_state()
        self.torques = compute_v4_torques(
            q, dq, self._v4_leg_targets, self._v4_wheel_targets,
            leg_kp=self.cfg.v4_leg_kp,
            leg_kd=self.cfg.v4_leg_kd,
            wheel_kd=self.cfg.v4_wheel_kd,
            wheel_effort_limit=self.cfg.v4_wheel_effort_limit,
        )
        self.robot.set_joint_effort_target(self.torques[:, :4], joint_ids=self._legs_act_idx)
        self.robot.set_joint_effort_target(self.torques[:, 4:6], joint_ids=self._wheel_idx)
        if self._gas_spring_enabled:
            self._apply_gas_spring_forces()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Keep V3 base bookkeeping alive (command resampling, ground estimate,
        # state machines) since V4 builds its own V5-style observation frame.
        self._update_ground_height_estimate()
        self._update_obs(True)
        self.command = self.command_generator.command.clone()
        self._on_command_updated()
        self._update_height_reward_airborne_state()

        data = self.robot.data
        q, dq, default = self._v4_policy_state()
        leg_pos = q[:, :4] - default[:, :4]
        contact = self._wheel_contact_flags() > 0.0
        mode, target, phase, phase_time = self._v4_task_fields(contact)
        compression, compression_rate = self._v4_spring_observation()
        v4_command = self._v4_command()
        proprio = torch.cat(
            (
                data.root_ang_vel_b * 0.5,
                data.projected_gravity_b,
                v4_command * v4_command.new_tensor([1.0, 1.0, 5.0]),
                leg_pos,
                dq * 0.1,
                self.last_actions,
            ), dim=-1
        )
        frame = torch.cat(
            (
                proprio,
                compression / self.cfg.v4_spring_compression_scale,
                compression_rate / self.cfg.v4_spring_rate_scale,
                mode,
                target,
                contact.to(proprio.dtype),
                phase,
                phase_time.clamp(max=5.0),
            ), dim=-1
        ).clamp(-100.0, 100.0)
        fresh = ~self._v4_history_initialized
        if fresh.any():
            self._v4_history[fresh] = frame[fresh, None, :].expand(-1, 5, -1)
            self._v4_history_initialized[fresh] = True
        else:
            self._v4_history[:, :-1] = self._v4_history[:, 1:].clone()
            self._v4_history[:, -1] = frame
        policy = self._v4_history.reshape(self.num_envs, V4_ACTOR_DIM)

        critic = torch.cat(
            (
                frame,
                data.root_lin_vel_b,
                (self._base_height() / 1.0).unsqueeze(-1),
                contact.to(frame.dtype),
                self.robot.data.joint_pos,
                self.robot.data.joint_vel,
                compression / self.cfg.v4_spring_compression_scale,
                compression_rate / self.cfg.v4_spring_rate_scale,
            ), dim=-1
        )
        critic = critic.clamp(-100.0, 100.0)
        validate_v4_shapes(policy, critic, self._actions, int(self.cfg.state_space))
        (
            self._obs_raw_policy_max_abs,
            self._obs_raw_policy_has_nonfinite,
        ) = self._per_env_max_abs_from_blocks(
            {"policy": frame}, num_envs=self.num_envs, device=self.device
        )
        (
            self._obs_raw_critic_max_abs,
            self._obs_raw_critic_has_nonfinite,
        ) = self._per_env_max_abs_from_blocks(
            {"critic": critic}, num_envs=self.num_envs, device=self.device
        )
        return {"policy": policy, "critic": critic}

    def _get_rewards(self) -> torch.Tensor:
        data = self.robot.data
        height = self._base_height()
        cmd = self._v4_command()
        contact = self._wheel_contact_flags()
        contact_gate = torch.exp(-((1.0 - contact).clamp(min=0.0) / 0.1).square()).prod(dim=-1)
        velocity = 2.0 * contact_gate * torch.exp(
            -((data.root_lin_vel_b[:, 0] - cmd[:, 0]) / 0.5).square()
        )
        yaw = contact_gate * torch.exp(-((data.root_ang_vel_b[:, 2] - cmd[:, 1]) / 0.5).square())
        height_reward = 2.0 * torch.exp(-((height - cmd[:, 2]) / 0.03).square())
        if self._gas_spring_prev_length is None:
            compression = torch.zeros(self.num_envs, 2, device=self.device)
        else:
            compression = self._gas_spring_nominal_length.unsqueeze(0) - self._gas_spring_prev_length
        spring_fraction = (compression / 0.08).clamp(min=0.0)
        spring_margin = torch.relu(spring_fraction - 0.9).square().sum(dim=-1)
        slip = (self._wheel_slip() * contact).sum(dim=-1)
        phase_landing = (self._v4_phase == 3).to(data.root_lin_vel_b.dtype)
        reward = (
            velocity + yaw + height_reward
            - 4.0 * data.projected_gravity_b[:, :2].square().sum(-1)
            - 0.05 * data.root_ang_vel_b[:, :2].square().sum(-1)
            - 0.5 * data.root_lin_vel_b[:, 2].square()
            - 0.01 * (self._actions - self._previous_actions).square().sum(-1)
            - 0.02 * (self.torques / self.torques.new_tensor([40, 40, 40, 40, 3.8377, 3.8377])).square().sum(-1)
            - 2.0 * slip
            - 1.0 * spring_margin
            - 0.05 * phase_landing * (self._contact_magnitudes()[:, self._desired_contact_link_idx].sum(-1) / 125.0).square()
        ) * self.step_dt
        closure_bad = self._get_closure_bad()
        if closure_bad is not None:
            reward = reward + self.cfg.v4_closure_penalty * closure_bad.to(reward.dtype) * self.step_dt
        reward = reward + self.cfg.v4_termination_penalty * self.reset_terminated.to(reward.dtype)
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not hasattr(self, "_v4_history"):
            return
        self._v4_history[ids] = 0.0
        self._v4_history_initialized[ids] = False
        self._v4_prev_spring_compression[ids] = 0.0
        self._v4_phase[ids] = 0
        self._v4_phase_time[ids] = 0.0
