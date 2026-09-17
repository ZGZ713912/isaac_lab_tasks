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
#   - act 4：joint_leg_* 位置目标（腿级联 PID：外环位置 PI→速度指令，内环速度 PI→力矩，
#     手工 effort 下发，与部署同构）；
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
from isaaclab.utils.math import quat_rotate_inverse

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
        self._prev2_actions = torch.zeros_like(self.actions)
        self.leg_target = torch.zeros(self.num_envs, len(self._legs_idx), device=self.device)
        self.q_cmd = torch.full(
            (self.num_envs,), self.cfg.default_q_cmd, device=self.device
        )
        self.cmd_buf = torch.zeros(self.num_envs, 3, device=self.device)
        self.cmd_timer = torch.zeros(self.num_envs, device=self.device)
        self._prev_joint_vel = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._prev_leg_torque = torch.zeros(self.num_envs, len(self._legs_idx), device=self.device)
        self._prev_root_ang_vel_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_root_lin_vel_z = torch.zeros(self.num_envs, device=self.device)

        # ---- 腿级联 PID 状态（外环/内环积分器 + 上一步速度指令）----
        self._leg_outer_int = torch.zeros(self.num_envs, len(self._legs_idx), device=self.device)
        self._leg_inner_int = torch.zeros_like(self._leg_outer_int)
        self._leg_vel_cmd = torch.zeros_like(self._leg_outer_int)

        # ---- 分层 spawn / 方向性指标 ----
        self._reset_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._n_dir_bins = max(1, int(self.cfg.spawn_dir_bins))
        self._dir_az_coverage = torch.zeros(self._n_dir_bins, device=self.device)
        self._dir_contact_ema = torch.zeros(self._n_dir_bins, device=self.device)
        self._dir_balance_ema = torch.zeros(self._n_dir_bins, device=self.device)
        self._dir_trackq_ema = torch.zeros(self._n_dir_bins, device=self.device)
        self._dir_up_frac_ema = torch.zeros((), device=self.device)
        self._dir_down_frac_ema = torch.zeros((), device=self.device)
        self._dir_flat_frac_ema = torch.zeros((), device=self.device)
        self._airborne_frac_ema = torch.zeros((), device=self.device)
        self._dir_ema_alpha = 0.01

        # 单轮目标载荷：默认按整车质量 1/4 估算
        self._wheel_load_target = (
            float(self.cfg.wheel_load_target)
            if float(self.cfg.wheel_load_target) > 0.0
            else self.cfg.chassis_total_mass * 9.81 / 4.0
        )

        self.episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device) for name in self.cfg.rewards
        }
        self.extras.setdefault("log", {})

        # ---- 周期坡面地形：识别 + 解析求高（与地形生成同源）----
        self._setup_periodic_profile()

    def _setup_periodic_profile(self) -> None:
        """识别周期坡面地形，构建 torch 坡角表与剖面坐标偏移。"""
        gen = getattr(self.cfg.terrain, "terrain_generator", None)
        sub = None
        if gen is not None and getattr(gen, "sub_terrains", None):
            sub = gen.sub_terrains.get("periodic_slope")
        self._periodic = sub is not None
        self._period_seg = 0.0
        self._slope_angle_table = None
        self._profile_x_offset = 0.0
        self._base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not self._periodic:
            return
        self._period_seg = float(sub.segment_length)
        size = gen.size
        num_periods = int(math.ceil(float(size[0]) / (4.0 * self._period_seg))) + 2
        self._slope_angle_table = du.build_periodic_slope_angle_table(
            self._period_seg, tuple(sub.angle_range), int(sub.angle_seed), num_periods, self.device
        )
        # 地形网格以世界原点为中心（[-size/2, size/2]）；剖面坐标 p = world_x + size/2。
        # 出生网格 scene.env_origins 本身即以原点为中心，故无需再偏移。
        self._profile_x_offset = float(size[0]) * 0.5

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
        """车体原点相对脚下地面的高度。

        周期坡面地形：`root_z − 解析地面高度(world_x)`；否则 `root_z − env_origin_z`。
        """
        z = self.robot.data.root_pos_w[:, 2]
        if self._periodic:
            ground = du.periodic_slope_height_torch(
                self.robot.data.root_pos_w[:, 0] + self._profile_x_offset,
                self._period_seg,
                self._slope_angle_table,
            )
            return z - ground
        return z - self.scene.env_origins[:, 2]

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
        self._prev2_actions.copy_(self.last_actions)  # a_{t-2}
        self.last_actions.copy_(self.actions)  # a_{t-1}，供 action_rate / action_rate2 用
        self.actions = actions.clone()  # a_t

        # 命令重采样（external_cmd_override=True 时由键盘等外部源写入，跳过重采样）
        if not self.cfg.external_cmd_override:
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
        if self.cfg.use_leg_cascade_pid:
            leg_tau = self._leg_cascade_torque(joint_pos, joint_vel)
        else:
            # 单环位置 PD 回退（与旧部署 kp/kd 一致）
            leg_tau = self.cfg.leg_stiffness * (self.leg_target - joint_pos[:, self._legs_idx]) \
                - self.cfg.leg_damping * joint_vel[:, self._legs_idx]
        torques = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        torques[:, self._legs_idx] = torch.clamp(
            leg_tau, -self.cfg.max_leg_torque, self.cfg.max_leg_torque
        )
        # wheel_set / upper_leg：mimic 硬约束跟随，不加力矩；wheels：零驱动
        self.robot.set_joint_effort_target(torques)
        self._apply_chassis_servo()

    def _leg_cascade_torque(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """腿级联 PID：外环位置 PI→速度指令，内环速度 PI→力矩（100 Hz 前向欧拉）。

        积分器均做限幅抗饱和；外环输出再经速度限幅，终力矩由 _apply_action 统一限幅。
        """
        q = joint_pos[:, self._legs_idx]
        qd = joint_vel[:, self._legs_idx]

        # 外环：位置误差 → 速度指令
        e_q = self.leg_target - q
        self._leg_outer_int += e_q * self.step_dt
        self._leg_outer_int.clamp_(-self.cfg.leg_outer_int_limit, self.cfg.leg_outer_int_limit)
        vel_cmd = self.cfg.leg_outer_kp * e_q + self.cfg.leg_outer_ki * self._leg_outer_int
        vel_cmd = vel_cmd.clamp(-self.cfg.leg_vel_cmd_limit, self.cfg.leg_vel_cmd_limit)
        self._leg_vel_cmd.copy_(vel_cmd)

        # 内环：速度误差 → 力矩
        e_v = vel_cmd - qd
        self._leg_inner_int += e_v * self.step_dt
        self._leg_inner_int.clamp_(-self.cfg.leg_inner_int_limit, self.cfg.leg_inner_int_limit)
        return self.cfg.leg_inner_kp * e_v + self.cfg.leg_inner_ki * self._leg_inner_int

    def _terrain_normal_w(self) -> torch.Tensor:
        """脚下地面在世界系的外法向，形状 (N,3)。

        周期坡面按解析梯度 `g = dh/dx` → `n = normalize([-g, 0, 1])`；否则取竖直向上。
        """
        n = torch.zeros(self.num_envs, 3, device=self.device)
        n[:, 2] = 1.0
        if self._periodic:
            p = self.robot.data.root_pos_w[:, 0] + self._profile_x_offset
            eps = 0.05
            h_fwd = du.periodic_slope_height_torch(p + eps, self._period_seg, self._slope_angle_table)
            h_bwd = du.periodic_slope_height_torch(p - eps, self._period_seg, self._slope_angle_table)
            n[:, 0] = -(h_fwd - h_bwd) / (2.0 * eps)
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return n

    def _apply_chassis_servo(self) -> None:
        """外部底盘速度伺服：对 base 施车身系力/力矩跟踪 (vx,vy,ωz)。

        球体碰撞轮不产生牵引力，故“运动工况”由此外部伺服代表。为忠实于地面耦合，
        期望力先投影到脚下地面切平面（去掉法向/抬升分量），再按库仑牵引上限
        `|F| ≤ μ·N_total` 限幅——离地时 N_total→0，力自动归零，杜绝“伺服托举悬空”。
        偏航力矩同理按 `μ·N_total·L` 限幅。
        """
        if not self.cfg.enable_chassis_servo:
            return
        v = self.robot.data.root_lin_vel_b
        w = self.robot.data.root_ang_vel_b
        fx = self.cfg.chassis_total_mass * self.cfg.chassis_servo_kp_lin * (self.cmd_buf[:, 0] - v[:, 0])
        fy = self.cfg.chassis_total_mass * self.cfg.chassis_servo_kp_lin * (self.cmd_buf[:, 1] - v[:, 1])
        tz = self.cfg.chassis_yaw_inertia * self.cfg.chassis_servo_kp_yaw * (self.cmd_buf[:, 2] - w[:, 2])
        fx = fx.clamp(-self.cfg.chassis_servo_max_force, self.cfg.chassis_servo_max_force)
        fy = fy.clamp(-self.cfg.chassis_servo_max_force, self.cfg.chassis_servo_max_force)

        # 期望车身系力 → 投影到地面切平面（n_b = 地面法向在车身系）
        force_b = torch.zeros(self.num_envs, 3, device=self.device)
        force_b[:, 0] = fx
        force_b[:, 1] = fy
        n_b = quat_rotate_inverse(self.robot.data.root_link_quat_w, self._terrain_normal_w())
        force_t = force_b - (force_b * n_b).sum(dim=-1, keepdim=True) * n_b

        # 库仑牵引限幅：地面最多传递 μ·N_total
        n_total = self.wheel_contact_forces.sum(dim=-1)
        f_cap = self.cfg.chassis_servo_friction_coeff * n_total
        scale = torch.clamp(f_cap / force_t.norm(dim=-1).clamp_min(1.0e-6), max=1.0)
        force_t = force_t * scale.unsqueeze(-1)

        tz_cap = torch.minimum(
            torch.full_like(n_total, self.cfg.chassis_servo_max_torque),
            self.cfg.chassis_servo_friction_coeff * n_total * du.OMNI_YAW_COEFF,
        )
        tz = tz.clamp(-tz_cap, tz_cap)

        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        forces[:, 0, :] = force_t
        torques[:, 0, 2] = tz
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
        # 底盘触地两阶段：前 N 轮软惩罚（不终止），之后死亡
        self._base_contact = base_contact
        iter_now = self.common_step_counter // max(1, self.cfg.training_progress_steps_per_iteration)
        base_contact_death = base_contact & (iter_now >= self.cfg.base_contact_death_after_iterations)
        terminated = base_contact_death | orientation_term | base_low_term | tunnel_term | nan_term
        time_out = self.episode_length_buf >= self.max_episode_length
        # 越界（跑出所属单位格）→ 按 time_out 重置
        if self._periodic and self.cfg.boundary_reset_enabled:
            origin = self.scene.env_origins
            half = max(0.0, 0.5 * float(self.cfg.scene.env_spacing) - self.cfg.boundary_reset_margin)
            dx = (self.robot.data.root_pos_w[:, 0] - origin[:, 0]).abs()
            dy = (self.robot.data.root_pos_w[:, 1] - origin[:, 1]).abs()
            time_out = time_out | (dx > half) | (dy > half)
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

        # 1) 四轮触地门控（20N）+ 目标载荷分布 + 归一化均力
        contact = torch.clamp(forces / self.cfg.desired_contact_force_threshold, 0.0, 1.0)
        terms["four_wheel_contact"] = contact.mean(dim=-1)
        terms["wheel_load_distribution"] = torch.exp(
            -torch.square(forces - self._wheel_load_target) / self.cfg.wheel_load_sigma
        ).mean(dim=-1)
        force_mean = forces.mean(dim=-1, keepdim=True)
        force_var = ((forces - force_mean) ** 2).mean(dim=-1)
        balance_denom = (
            self.cfg.wheel_force_balance_sigma_rel * torch.square(force_mean.squeeze(-1)) + 1.0
        )
        terms["wheel_force_balance"] = torch.exp(-force_var / balance_denom)

        # 2) 车身水平（接地门控：防“翘轮换水平”）
        contact_gate = (
            terms["four_wheel_contact"]
            if self.cfg.gate_orientation_by_contact
            else torch.ones_like(terms["four_wheel_contact"])
        )
        terms["flat_orientation_x_exp"] = contact_gate * torch.exp(
            -torch.square(pgb[:, 1]) / self.cfg.orientation_x_exp_sigma
        )
        terms["flat_orientation_y_exp"] = contact_gate * torch.exp(
            -torch.square(pgb[:, 0]) / self.cfg.orientation_y_exp_sigma
        )

        # 3) 基准腿角跟踪
        q_err = self.robot.data.joint_pos[:, self._legs_idx] - self.q_cmd.unsqueeze(-1)
        terms["track_q_cmd_exp"] = torch.exp(
            -torch.square(q_err).mean(dim=-1) / self.cfg.q_track_sigma
        )
        # 3b) 低模式软偏好贴地（仅低模式生效：q_cmd 大 = 车低；不强制某条腿）
        low_mask = (self.q_cmd > self.cfg.low_mode_q_threshold).float()
        if self._periodic:
            # 坡上需要抬身过坡，贴地偏好只在平路段生效
            low_mask = low_mask * du.periodic_slope_flat_mask(
                self.robot.data.root_pos_w[:, 0], self._period_seg
            ).float()
        excess = torch.clamp(self.base_height - du.H_LOW, min=0.0)
        terms["low_height_pref"] = low_mask * torch.exp(
            -torch.square(excess) / self.cfg.low_height_sigma
        )

        # 3c) 底盘触地（软惩罚；≥N 轮后同时判死亡，见 _get_dones）
        terms["base_contact"] = self._base_contact.float()

        # 4) 常规惩罚
        terms["torques"] = torch.sum(torch.square(applied_torque[:, self._legs_idx]), dim=-1)
        terms["action_rate"] = torch.sum(torch.square(self.actions - self.last_actions), dim=-1)
        terms["action_rate2"] = torch.sum(
            torch.square(self.actions - 2.0 * self.last_actions + self._prev2_actions), dim=-1
        )
        terms["leg_joint_vel"] = torch.sum(torch.square(joint_vel[:, self._legs_idx]), dim=-1)
        terms["leg_joint_acc"] = torch.sum(
            torch.square(
                (joint_vel[:, self._legs_idx] - self._prev_joint_vel[:, self._legs_idx])
                / self.step_dt
            ),
            dim=-1,
        )
        leg_torque = applied_torque[:, self._legs_idx]
        terms["leg_torque_rate"] = torch.sum(
            torch.square(leg_torque - self._prev_leg_torque), dim=-1
        )
        root_ang_vel_xy = self.robot.data.root_ang_vel_b[:, :2]
        terms["base_ang_acc"] = torch.sum(
            torch.square((root_ang_vel_xy - self._prev_root_ang_vel_xy) / self.step_dt), dim=-1
        )
        root_lin_vel_z = self.robot.data.root_lin_vel_b[:, 2]
        terms["base_lin_acc_z"] = torch.square(
            (root_lin_vel_z - self._prev_root_lin_vel_z) / self.step_dt
        )
        terms["ang_vel_xy"] = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=-1)
        terms["lin_vel_z"] = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        undesired = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self._legs_contact_idx, :], dim=-1
        )
        terms["undesired_contact"] = torch.any(
            undesired > self.cfg.undesired_contact_force_threshold, dim=-1
        ).float()

        # 5) 方向性指标（覆盖率 + 分方位效果 EMA）
        self._update_direction_metrics(terms, pgb)

        reward = torch.zeros(self.num_envs, device=self.device)
        for name, weight in self.cfg.rewards.items():
            reward += weight * terms[name]
            self.episode_sums[name] += terms[name]
        self._prev_joint_vel.copy_(joint_vel)
        self._prev_leg_torque.copy_(leg_torque)
        self._prev_root_ang_vel_xy.copy_(root_ang_vel_xy)
        self._prev_root_lin_vel_z.copy_(root_lin_vel_z)
        return reward

    def _update_direction_metrics(
        self, terms: dict[str, torch.Tensor], pgb: torch.Tensor
    ) -> None:
        """更新车体系坡度方位覆盖率与分方位效果 EMA，供 extras["log"] 验收方向均匀性。

        - 车体系坡度方位由重力水平投影 az=atan2(pgb_y, pgb_x) 得到；
        - 上/下/平由解析坡面在 root_x 的局部梯度符号判定，平坦段不进入方位统计；
        - 覆盖率与分方位效果只在坡段（|g|>tol）上统计并归一化。
        """
        alpha = self._dir_ema_alpha
        n_bins = self._n_dir_bins

        # 悬空占比（四轮法向力之和 < 阈值）
        n_total = self.wheel_contact_forces.sum(dim=-1)
        airborne = (n_total < self.cfg.airborne_force_threshold).float().mean()
        self._airborne_frac_ema = (1.0 - alpha) * self._airborne_frac_ema + alpha * airborne

        if self._periodic:
            p = self.robot.data.root_pos_w[:, 0] + self._profile_x_offset
            eps = 0.05
            h_fwd = du.periodic_slope_height_torch(p + eps, self._period_seg, self._slope_angle_table)
            h_bwd = du.periodic_slope_height_torch(p - eps, self._period_seg, self._slope_angle_table)
            grad = (h_fwd - h_bwd) / (2.0 * eps)
            tol = 1.0e-3
            is_up = grad > tol
            is_down = grad < -tol
        else:
            is_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            is_down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        is_flat = ~(is_up | is_down)
        self._dir_up_frac_ema = (1.0 - alpha) * self._dir_up_frac_ema + alpha * is_up.float().mean()
        self._dir_down_frac_ema = (
            (1.0 - alpha) * self._dir_down_frac_ema + alpha * is_down.float().mean()
        )
        self._dir_flat_frac_ema = (
            (1.0 - alpha) * self._dir_flat_frac_ema + alpha * is_flat.float().mean()
        )

        # 车体系坡度方位（仅坡段参与统计）
        az = torch.atan2(pgb[:, 1], pgb[:, 0])
        bins = torch.floor((az + math.pi) / (2.0 * math.pi) * n_bins).long().clamp_(0, n_bins - 1)
        onehot = torch.nn.functional.one_hot(bins, n_bins).to(self._dir_az_coverage.dtype)
        slope_mask = (is_up | is_down).to(onehot.dtype)
        total = slope_mask.sum()
        if float(total) > 0.0:
            coverage_step = (onehot * slope_mask.unsqueeze(-1)).sum(dim=0) / total
            self._dir_az_coverage = (1.0 - alpha) * self._dir_az_coverage + alpha * coverage_step

        def _bin_ema(ema: torch.Tensor, value: torch.Tensor) -> None:
            counts = (onehot * slope_mask.unsqueeze(-1)).sum(dim=0)
            valid = counts > 0.0
            summed = (onehot * (value * slope_mask).unsqueeze(-1)).sum(dim=0)
            mean_per_bin = summed / counts.clamp_min(1.0)
            ema[valid] = (1.0 - alpha) * ema[valid] + alpha * mean_per_bin[valid]

        _bin_ema(self._dir_contact_ema, terms["four_wheel_contact"])
        _bin_ema(self._dir_balance_ema, terms["wheel_force_balance"])
        _bin_ema(self._dir_trackq_ema, terms["track_q_cmd_exp"])

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        # 父类通用重置：scene.reset（清传感器 buffer）+ episode_length_buf 归零 + 事件/噪声重置。
        # 必须调用：否则 episode_length_buf 永不归零，超过 max_episode_length 后每步都判 time_out，
        # 机器人被每步瞬移+随机 yaw，表现为疯狂旋转与上下弹跳。
        super()._reset_idx(env_ids)

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

        # 根状态：分层/循环方向 spawn（周期坡面用解析地面高度），10cm 下落
        origin = self.scene.env_origins[env_ids]
        root_pos = origin.clone()
        if self._periodic:
            half = max(0.0, 0.5 * float(self.cfg.scene.env_spacing) - self.cfg.boundary_reset_margin)
            if self.cfg.spawn_dir_stratify:
                n_bins = self._n_dir_bins
                self._reset_count[env_ids] += 1
                # combo 在 (方位 bin × 上/下坡) 上循环，保证 batch 内精确均匀且逐次旋转
                combo = (env_ids + self._reset_count[env_ids]) % (2 * n_bins)
                sign = combo % 2  # 0=上坡, 1=下坡
                az_bin = combo // 2
                step = 2.0 * math.pi / n_bins
                jitter = (
                    (torch.rand(n, device=device) - 0.5) * step
                    if self.cfg.spawn_dir_jitter
                    else torch.zeros(n, device=device)
                )
                psi = az_bin.to(torch.float32) * step + jitter  # 车体系上坡方位
                yaw = torch.where(sign == 0, -psi, math.pi - psi)
                # 相位：上坡 [0,L)，下坡 [2L,3L)；关相位分层则整周期随机
                seg = self._period_seg
                period = 4.0 * seg
                if self.cfg.spawn_phase_stratify:
                    u = torch.rand(n, device=device)
                    s_phase = torch.where(sign == 0, u * seg, 2.0 * seg + u * seg)
                else:
                    s_phase = torch.rand(n, device=device) * period
                # 在本 env 单元内选合法周期（k0-1/k0/k0+1），使 world_x 靠近单元中心
                k0 = torch.floor((origin[:, 0] + self._profile_x_offset) / period).long()
                offsets = torch.tensor([-1, 0, 1], device=device).unsqueeze(0)
                cand = (k0.unsqueeze(1) + offsets).clamp_(0, self._slope_angle_table.numel() - 1)
                cand_x = cand.to(root_pos.dtype) * period + s_phase.unsqueeze(1) - self._profile_x_offset
                dx = (cand_x - origin[:, 0:1]).abs()
                best = torch.argmin(torch.where(dx <= half, dx, dx + 1.0e6), dim=1)
                best_x = cand_x[torch.arange(n, device=device), best]
                # 单元落在坡面表范围外时回退为单元内随机位置（保底不越界）
                fallback_x = origin[:, 0] + (torch.rand(n, device=device) * 2.0 - 1.0) * half
                root_pos[:, 0] = torch.where((best_x - origin[:, 0]).abs() <= half, best_x, fallback_x)
                root_pos[:, 1] = origin[:, 1] + (torch.rand(n, device=device) * 2.0 - 1.0) * half
            else:
                root_pos[:, 0] += (torch.rand(n, device=device) * 2.0 - 1.0) * half
                root_pos[:, 1] += (torch.rand(n, device=device) * 2.0 - 1.0) * half
                yaw = torch.rand(n, device=device) * 2 * math.pi - math.pi
            ground = du.periodic_slope_height_torch(
                root_pos[:, 0] + self._profile_x_offset, self._period_seg, self._slope_angle_table
            )
            root_pos[:, 2] = origin[:, 2] + ground
        else:
            yaw = torch.rand(n, device=device) * 2 * math.pi - math.pi
        root_pos[:, 2] += du.q_to_base_height(self.q_cmd[env_ids]) + self.cfg.reset_height_buffer
        root_pose = torch.zeros(n, 7, device=device)
        root_pose[:, :3] = root_pos
        root_pose[:, 3] = torch.cos(yaw / 2)
        root_pose[:, 6] = torch.sin(yaw / 2)
        root_vel6 = torch.zeros(n, 6, device=device)
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel6, env_ids=env_ids)

        # buffers
        if self.cfg.external_cmd_override:
            # 外部（键盘）控制命令：重置时归零，之后由外部写入
            self.cmd_buf[env_ids] = 0.0
        else:
            self._resample_commands(env_ids)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self._prev2_actions[env_ids] = 0.0
        self.leg_target[env_ids] = self.q_cmd[env_ids].unsqueeze(-1)
        self._prev_joint_vel[env_ids] = 0.0
        self._prev_leg_torque[env_ids] = 0.0
        self._prev_root_ang_vel_xy[env_ids] = 0.0
        self._prev_root_lin_vel_z[env_ids] = 0.0
        # 级联 PID 积分器/速度指令：teleport 后必须清零，防积分残留
        self._leg_outer_int[env_ids] = 0.0
        self._leg_inner_int[env_ids] = 0.0
        self._leg_vel_cmd[env_ids] = 0.0

        # 日志
        self.extras["log"] = {}
        for name, sums in self.episode_sums.items():
            self.extras["log"][f"episode/{name}"] = sums[env_ids].mean().item()
            sums[env_ids] = 0.0
        self._log_direction_metrics()

    def _log_direction_metrics(self) -> None:
        """把方向覆盖率与分方位效果 EMA 写入 extras["log"]（验收各向均匀性）。"""
        log = self.extras["log"]
        for i in range(self._n_dir_bins):
            log[f"dir/az_bin{i}"] = self._dir_az_coverage[i].item()
            log[f"dir/contact_bin{i}"] = self._dir_contact_ema[i].item()
            log[f"dir/balance_bin{i}"] = self._dir_balance_ema[i].item()
            log[f"dir/trackq_bin{i}"] = self._dir_trackq_ema[i].item()
        log["dir/slope_up_frac"] = self._dir_up_frac_ema.item()
        log["dir/slope_down_frac"] = self._dir_down_frac_ema.item()
        log["dir/slope_flat_frac"] = self._dir_flat_frac_ema.item()
        log["contact/airborne_frac"] = self._airborne_frac_ema.item()
