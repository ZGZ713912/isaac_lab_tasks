# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# DeformableSuspensionEnv —— 平四变形底盘主动悬挂（单层 DirectRLEnv，照 wheelbipe 惯例）。
#
# 设计要点：
#   - act 4：joint_leg_* 位置 PD 目标（手工 effort PD，kp=200/kd=4；与部署同构）；
#     joint_wheel_set_* / joint_upper_leg_* 由 URDF <mimic> 硬约束跟随，不施加力矩。
#   - obs 26：q_cmd | cmd(vx,vy,ωz) | ang_vel_b | proj_grav_b | leg_pos(绝对) |
#     leg_vel | leg_torque | act；critic 追加 lin_vel_b、真实车高、四轮接触力。
#   - 奖励：四轮触地 + 法向力均衡 + 车身水平 + 基准角跟踪 + 低模式贴地偏好 + 常规。
#   - 运动：球体碰撞轮无牵引力 → 由外部底盘速度伺服实现（首版静态关闭）。
# =============================================================================

from __future__ import annotations

import math
import re

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from agent_tasks.direct.deformable_suspension import cfg_utils as du
from agent_tasks.direct.deformable_suspension.env_cfg import DeformableSuspensionBaseEnvCfg

# policy 观测块（名称, 维度），顺序必须与 _get_observations 拼接一致
_OBS_BLOCKS = (
    ("q_cmd", 1),
    ("cmd", 3),
    ("ang_vel", 3),
    ("gravity", 3),
    ("leg_pos", 4),
    ("leg_vel", 4),
    ("leg_torque", 4),
    ("act", 4),
)


class DeformableSuspensionEnv(DirectRLEnv):
    cfg: DeformableSuspensionBaseEnvCfg

    def __init__(self, cfg: DeformableSuspensionBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- 关节索引（joint_wheel_(?!set_).* 排除轮架从动关节）----
        self._legs_idx, _ = self.robot.find_joints("joint_leg_.*")
        self._ws_idx, _ = self.robot.find_joints("joint_wheel_set_.*")
        self._upper_idx, _ = self.robot.find_joints("joint_upper_leg_.*")
        self._wheels_idx, _ = self.robot.find_joints("joint_wheel_(?!set_).*")
        self._num_joints = self.robot.num_joints

        # ---- 接触索引（接触传感器 body 顺序与 robot 不一致，按名映射）----
        self._base_contact_idx = self._find_contact_sensor_indices("base_link")
        self._legs_contact_idx = self._find_contact_sensor_indices(
            ["leg_.*", "wheel_set_.*", "upper_leg_.*"]
        )
        self._wheels_contact_idx = self._find_contact_sensor_indices("wheel_(?!set_).*")

        # ---- 底盘速度伺服作用体 ----
        base_body_ids, _ = self.robot.find_bodies("base_link")
        self._base_body_id = base_body_ids

        self._validate_bookkeeping()

        # ---- buffers ----
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.leg_target = torch.zeros(self.num_envs, len(self._legs_idx), device=self.device)
        self.q_cmd = torch.full(
            (self.num_envs,), self.cfg.default_q_cmd, device=self.device
        )
        self.cmd_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.cmd_timer = torch.zeros(self.num_envs, device=self.device)
        self._prev_joint_vel = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self.cfg.rewards
        }
        self.extras.setdefault("log", {})

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _find_contact_sensor_indices(self, body_names_expr: str | list[str]) -> list[int]:
        if isinstance(body_names_expr, str):
            body_names_expr = [body_names_expr]
        indices: list[int] = []
        for expr in body_names_expr:
            pattern = re.compile(expr)
            for i, name in enumerate(self.contact_sensor.body_names):
                if pattern.match(name):
                    indices.append(i)
        return sorted(set(indices))

    def _validate_bookkeeping(self) -> None:
        """启动自检：关节/接触体命名与数量，防静默漏配。"""
        assert len(self._legs_idx) == 4, f"expected 4 leg joints, got {self._legs_idx}"
        assert len(self._ws_idx) == 4, f"expected 4 wheel_set joints, got {self._ws_idx}"
        assert len(self._upper_idx) == 4, f"expected 4 upper_leg joints, got {self._upper_idx}"
        assert len(self._wheels_idx) == 4, f"expected 4 wheel joints, got {self._wheels_idx}"
        assert len(self._wheels_contact_idx) == 4, (
            f"expected 4 wheel contact bodies, got {self._wheels_contact_idx}"
        )
        assert len(self._base_contact_idx) >= 1, "base_link contact body not found"

    @property
    def base_height(self) -> torch.Tensor:
        """车体原点相对地形高度（root z − env origin z）。"""
        return self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    @property
    def wheel_contact_forces(self) -> torch.Tensor:
        """四轮法向接触力模长（N），形状 (N,4)。"""
        return torch.norm(
            self.contact_sensor.data.net_forces_w[:, self._wheels_contact_idx, :], dim=-1
        )

    def _clip_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        i = 0
        for name, dim in _OBS_BLOCKS:
            lo, hi = du.OBS_CLIP[name]
            obs[:, i : i + dim] = obs[:, i : i + dim].clamp(lo, hi)
            i += dim
        return obs

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        device = self.device
        for axis, rng in enumerate(
            (
                self.cfg.cmd_lin_vel_x_range,
                self.cfg.cmd_lin_vel_y_range,
                self.cfg.cmd_ang_vel_z_range,
            )
        ):
            self.cmd_buf[env_ids, axis] = torch.rand(n, device=device) * (rng[1] - rng[0]) + rng[0]
        self.cmd_timer[env_ids] = torch.rand(n, device=device) * (
            self.cfg.cmd_resample_time_range[1] - self.cfg.cmd_resample_time_range[0]
        ) + self.cfg.cmd_resample_time_range[0]

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # action
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.last_actions.copy_(self.actions)  # 先存上一步，供 action_rate 用
        self.actions = actions.clone()

        # 命令重采样（首版范围全 0，仍保留机制）
        self.cmd_timer -= self.step_dt
        resample_ids = (self.cmd_timer <= 0.0).nonzero(as_tuple=False).squeeze(-1)
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)

        # 腿位置 PD 目标：q_target = q_cmd + action_scale·a（与部署同构）
        self.leg_target = torch.clamp(
            self.q_cmd.unsqueeze(-1) + self.cfg.leg_action_scale * self.actions,
            du.LEG_LOWER_LIMIT,
            du.LEG_UPPER_LIMIT,
        )

    def _apply_action(self) -> None:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        # 腿位置 PD（与部署 kp/kd 一致）
        leg_pd = self.cfg.leg_stiffness * (self.leg_target - joint_pos[:, self._legs_idx]) \
            - self.cfg.leg_damping * joint_vel[:, self._legs_idx]
        torques = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        torques[:, self._legs_idx] = torch.clamp(
            leg_pd, -self.cfg.max_leg_torque, self.cfg.max_leg_torque
        )
        # wheel_set / upper_leg：mimic 硬约束跟随，不加力矩；wheels：零驱动
        self.robot.set_joint_effort_target(torques)
        self._apply_chassis_servo()

    def _apply_chassis_servo(self) -> None:
        """外部底盘速度伺服：对 base 施车身系力/力矩跟踪 (vx,vy,ωz)。

        球体碰撞轮不产生牵引力，故“运动工况”由此外部伺服代表（首版关闭）。
        """
        if not self.cfg.enable_chassis_servo:
            return
        v = self.robot.data.root_lin_vel_b
        w = self.robot.data.root_ang_vel_b
        fx = self.cfg.chassis_total_mass * self.cfg.chassis_servo_kp_lin * (self.cmd_buf[:, 0] - v[:, 0])
        fy = self.cfg.chassis_total_mass * self.cfg.chassis_servo_kp_lin * (self.cmd_buf[:, 1] - v[:, 1])
        tz = self.cfg.chassis_yaw_inertia * self.cfg.chassis_servo_kp_yaw * (self.cmd_buf[:, 2] - w[:, 2])
        forces = torch.zeros(self.num_envs, 3, device=self.device)
        torques = torch.zeros(self.num_envs, 3, device=self.device)
        forces[:, 0] = fx.clamp(-self.cfg.chassis_servo_max_force, self.cfg.chassis_servo_max_force)
        forces[:, 1] = fy.clamp(-self.cfg.chassis_servo_max_force, self.cfg.chassis_servo_max_force)
        torques[:, 2] = tz.clamp(-self.cfg.chassis_servo_max_torque, self.cfg.chassis_servo_max_torque)
        self.robot.set_external_force_and_torque(forces, torques, body_ids=self._base_body_id)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        leg_pos = self.robot.data.joint_pos[:, self._legs_idx]
        leg_vel = self.robot.data.joint_vel[:, self._legs_idx]
        leg_torque = self.robot.data.applied_torque[:, self._legs_idx]
        obs = torch.cat(
            [
                self.q_cmd.unsqueeze(-1) * self.cfg.q_cmd_scale,
                self.cmd_buf * self.cfg.cmd_scale,
                self.robot.data.root_ang_vel_b * self.cfg.ang_vel_scale,
                self.robot.data.projected_gravity_b,
                leg_pos * self.cfg.joint_pos_scale,  # 绝对腿角（非相对量）
                leg_vel * self.cfg.joint_vel_scale,
                leg_torque * self.cfg.joint_torque_scale,
                self.actions,
            ],
            dim=-1,
        )
        obs = self._clip_policy_obs(obs)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        # 特权观测（asymmetric critic）
        critic = torch.cat(
            [
                obs,
                self.robot.data.root_lin_vel_b,
                self.base_height.unsqueeze(-1),
                self.wheel_contact_forces,
            ],
            dim=-1,
        )
        critic = torch.nan_to_num(critic, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs, "critic": critic}

    # ------------------------------------------------------------------
    # termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_contact = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self._base_contact_idx, :], dim=-1
        ).max(dim=-1).values > 1.0
        pgb = self.robot.data.projected_gravity_b
        roll_lim = math.sin(math.radians(self.cfg.termination_roll_deg))
        pitch_lim = math.sin(math.radians(self.cfg.termination_pitch_deg))
        orientation_term = (pgb[:, 0].abs() > pitch_lim) | (pgb[:, 1].abs() > roll_lim)
        base_low_term = self.base_height < self.cfg.terminate_base_height_low
        tunnel_term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.terminate_body_top is not None:
            tunnel_term = (self.base_height + du.BODY_TOP_OFFSET) > self.cfg.terminate_body_top
        nan_term = (
            ~torch.isfinite(self.robot.data.joint_pos).all(dim=-1)
            | ~torch.isfinite(self.robot.data.root_lin_vel_b).all(dim=-1)
            | ~torch.isfinite(self.robot.data.root_ang_vel_b).all(dim=-1)
        )
        terminated = base_contact | orientation_term | base_low_term | tunnel_term | nan_term
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        pgb = self.robot.data.projected_gravity_b
        applied_torque = self.robot.data.applied_torque
        joint_vel = self.robot.data.joint_vel
        forces = self.wheel_contact_forces  # (N,4)

        terms: dict[str, torch.Tensor] = {}
        terms["alive"] = torch.ones(self.num_envs, device=self.device)
        terms["termination"] = self.reset_terminated.float()

        # 1) 四轮触地 + 均力
        contact = torch.clamp(forces / self.cfg.wheel_contact_force_threshold, 0.0, 1.0)
        terms["four_wheel_contact"] = contact.mean(dim=-1)
        force_mean = forces.mean(dim=-1, keepdim=True)
        force_var = ((forces - force_mean) ** 2).mean(dim=-1)
        terms["wheel_force_balance"] = torch.exp(-force_var / self.cfg.wheel_force_balance_sigma)

        # 2) 车身水平（严格水平 → σ 小、权重高）
        terms["flat_orientation_x_exp"] = torch.exp(
            -torch.square(pgb[:, 1]) / self.cfg.orientation_x_exp_sigma
        )
        terms["flat_orientation_y_exp"] = torch.exp(
            -torch.square(pgb[:, 0]) / self.cfg.orientation_y_exp_sigma
        )

        # 3) 基准腿角跟踪
        q_err = self.robot.data.joint_pos[:, self._legs_idx] - self.q_cmd.unsqueeze(-1)
        terms["track_q_cmd_exp"] = torch.exp(
            -torch.square(q_err).mean(dim=-1) / self.cfg.q_track_sigma
        )
        # 3b) 低模式软偏好贴地（仅低模式生效：q_cmd 大 = 车低；不强制某条腿）
        low_mask = (self.q_cmd > self.cfg.low_mode_q_threshold).float()
        excess = torch.clamp(self.base_height - du.H_LOW, min=0.0)
        terms["low_height_pref"] = low_mask * torch.exp(
            -torch.square(excess) / self.cfg.low_height_sigma
        )

        # 4) 常规惩罚
        terms["torques"] = torch.sum(torch.square(applied_torque[:, self._legs_idx]), dim=-1)
        terms["action_rate"] = torch.sum(torch.square(self.actions - self.last_actions), dim=-1)
        terms["leg_joint_vel"] = torch.sum(torch.square(joint_vel[:, self._legs_idx]), dim=-1)
        terms["leg_joint_acc"] = torch.sum(
            torch.square(
                (joint_vel[:, self._legs_idx] - self._prev_joint_vel[:, self._legs_idx])
                / self.step_dt
            ),
            dim=-1,
        )
        terms["ang_vel_xy"] = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=-1)
        terms["lin_vel_z"] = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        undesired = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self._legs_contact_idx, :], dim=-1
        )
        terms["undesired_contact"] = torch.any(
            undesired > self.cfg.undesired_contact_force_threshold, dim=-1
        ).float()

        reward = torch.zeros(self.num_envs, device=self.device)
        for name, weight in self.cfg.rewards.items():
            reward += weight * terms[name]
            self.episode_sums[name] += terms[name]
        self._prev_joint_vel.copy_(joint_vel)
        return reward

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        device = self.device

        # 基准角：首版两档离散；后续可连续
        if self.cfg.use_continuous_q_cmd:
            self.q_cmd[env_ids] = torch.rand(n, device=device) * (
                self.cfg.q_cmd_range[1] - self.cfg.q_cmd_range[0]
            ) + self.cfg.q_cmd_range[0]
        else:
            choices = torch.tensor(self.cfg.q_cmd_choices, device=device)
            idx = torch.randint(0, len(self.cfg.q_cmd_choices), (n,), device=device)
            self.q_cmd[env_ids] = choices[idx]

        # 关节：闭链一致位姿（leg=q_cmd, ws=q_cmd, upper=−q_cmd），轮子随机角
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._legs_idx] = self.q_cmd[env_ids].unsqueeze(-1)
        joint_pos[:, self._ws_idx] = self.q_cmd[env_ids].unsqueeze(-1)
        joint_pos[:, self._upper_idx] = -self.q_cmd[env_ids].unsqueeze(-1)
        joint_pos[:, self._wheels_idx] = (
            torch.rand(n, len(self._wheels_idx), device=device) * 4 * math.pi - 2 * math.pi
        )
        joint_vel = torch.zeros(n, self._num_joints, device=device)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 根状态：地形 origin + 标定车高 + 缓冲；随机 yaw；速度归零
        root_pos = self.scene.env_origins[env_ids].clone()
        root_pos[:, 2] += du.q_to_base_height(self.q_cmd[env_ids]) + self.cfg.reset_height_buffer
        yaw = torch.rand(n, device=device) * 2 * math.pi - math.pi
        root_pose = torch.zeros(n, 7, device=device)
        root_pose[:, :3] = root_pos
        root_pose[:, 3] = torch.cos(yaw / 2)
        root_pose[:, 6] = torch.sin(yaw / 2)
        root_vel6 = torch.zeros(n, 6, device=device)
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel6, env_ids=env_ids)

        # buffers
        self._resample_commands(env_ids)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.leg_target[env_ids] = self.q_cmd[env_ids].unsqueeze(-1)
        self._prev_joint_vel[env_ids] = 0.0

        # 日志
        self.extras["log"] = {}
        for name, sums in self.episode_sums.items():
            self.extras["log"][f"episode/{name}"] = sums[env_ids].mean().item()
            sums[env_ids] = 0.0
