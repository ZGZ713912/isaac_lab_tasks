# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# DeformableSuspensionEnv —— 变形底盘主动悬挂 DirectRLEnv
#
# 与 RMCS rmcs_rl 部署合同逐项同构（obs 22 / act 4）：
#   obs = cmd3 | height_cmd1 | ang_vel3(×0.5) | gravity3 |
#         leg_pos4(×1.0) | leg_vel4(×0.1) | act4
#   act = 4 腿关节位置 PD 目标（action_scale 0.25，kp=200 kd=4）
# 腿/轮架平四耦合用虚拟刚弹簧；轮子零驱动（自由滚动）。
# =============================================================================

from __future__ import annotations

import math
import re
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .env_cfg import DeformableSuspensionBaseEnvCfg


class DeformableSuspensionEnv(DirectRLEnv):
    cfg: DeformableSuspensionBaseEnvCfg

    def __init__(self, cfg: DeformableSuspensionBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- joint indices（按名解析，顺序无关）----
        self._legs_idx, _ = self.robot.find_joints("joint_leg_.*")
        self._ws_idx, _ = self.robot.find_joints("joint_wheel_set_.*")
        self._wheels_idx, _ = self.robot.find_joints("joint_wheel_.*")
        self._num_joints = self.robot.num_joints

        # ---- contact indices（接触传感器 body 顺序与 robot 不一致，需按名映射）----
        self._base_contact_idx = self._find_contact_sensor_indices("base_link")
        self._legs_contact_idx = self._find_contact_sensor_indices(["leg_.*", "wheel_set_.*"])
        self._wheels_contact_idx = self._find_contact_sensor_indices("wheel_.*")

        # ---- buffers ----
        self.height_cmd = torch.full(
            (self.num_envs,), self.cfg.default_height_cmd, dtype=torch.float, device=self.device
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, dtype=torch.float, device=self.device
        )
        self.leg_target = torch.zeros(
            self.num_envs, len(self._legs_idx), dtype=torch.float, device=self.device
        )
        self._prev_joint_vel = torch.zeros(
            self.num_envs, self._num_joints, dtype=torch.float, device=self.device
        )
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self.cfg.rewards
        }
        self.extras.setdefault("log", {})

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _find_contact_sensor_indices(self, body_names_expr: str | list[str]) -> list[int]:
        """按正则表达式在接触传感器 body_names 中寻找索引（解决 robot 索引不匹配问题）。"""
        if isinstance(body_names_expr, str):
            body_names_expr = [body_names_expr]
        indices = []
        for expr in body_names_expr:
            pattern = re.compile(expr)
            for i, name in enumerate(self.contact_sensor.body_names):
                if pattern.match(name):
                    indices.append(i)
        return sorted(set(indices))

    @property
    def base_height(self) -> torch.Tensor:
        """车体原点相对地形高度（root z − env origin z，粗糙地形下语义正确）。"""
        return self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    @property
    def wheel_contact_forces(self) -> torch.Tensor:
        return torch.norm(
            self.contact_sensor.data.net_forces_w[:, self._wheels_contact_idx, :], dim=-1
        )

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # robot
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        # contact sensor
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # action
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.last_actions.copy_(self.actions)
        # 腿位置 PD 目标：target = action_scale × a + default（与部署 RlController 同构）
        self.leg_target = torch.clamp(
            self.cfg.leg_action_scale * actions
            + self.robot.data.default_joint_pos[:, self._legs_idx],
            min=0.0,
            max=1.05,
        )

    def _apply_action(self) -> None:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        # 腿位置 PD（与部署 kp/kd 一致）
        leg_pd = self.cfg.leg_stiffness * (self.leg_target - joint_pos[:, self._legs_idx]) \
            - self.cfg.leg_damping * joint_vel[:, self._legs_idx]
        # 平四耦合虚拟弹簧：轮架跟随腿（θ_ws == θ_leg），反作用载荷加载到腿
        coupling = self.cfg.coupling_stiffness * (
            joint_pos[:, self._legs_idx] - joint_pos[:, self._ws_idx]
        ) + self.cfg.coupling_damping * (
            joint_vel[:, self._legs_idx] - joint_vel[:, self._ws_idx]
        )
        torques = torch.zeros(self.num_envs, self._num_joints, dtype=torch.float, device=self.device)
        torques[:, self._legs_idx] = torch.clamp(
            leg_pd - coupling, -self.cfg.max_leg_torque, self.cfg.max_leg_torque
        )
        torques[:, self._ws_idx] = torch.clamp(
            coupling, -self.cfg.max_leg_torque, self.cfg.max_leg_torque
        )
        # wheels: 零驱动（自由滚动）
        self.robot.set_joint_effort_target(torques)

    # ------------------------------------------------------------------
    # observations（合同与部署逐项同构）
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos[:, self._legs_idx] \
            - self.robot.data.default_joint_pos[:, self._legs_idx]
        joint_vel = self.robot.data.joint_vel[:, self._legs_idx]
        obs = torch.cat(
            [
                torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device),  # cmd3（预留）
                self.height_cmd.unsqueeze(-1) * self.cfg.height_scale,  # 1
                self.robot.data.root_ang_vel_b * self.cfg.ang_vel_scale,  # 3
                self.robot.data.projected_gravity_b,  # 3
                joint_pos * self.cfg.joint_pos_scale,  # 4
                joint_vel * self.cfg.joint_vel_scale,  # 4
                self.actions,  # 4
            ],
            dim=-1,
        )
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        # 特权观测（asymmetric critic）：+ lin_vel3 + 车体相对高度1
        critic = torch.cat(
            [
                obs,
                self.robot.data.root_lin_vel_b,
                self.base_height.unsqueeze(-1) * self.cfg.height_scale,
            ],
            dim=-1,
        )
        return {"policy": obs, "critic": critic}

    # ------------------------------------------------------------------
    # termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_contact = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self._base_contact_idx, :], dim=-1
        ).squeeze(-1) > 1.0
        pgb = self.robot.data.projected_gravity_b
        roll_lim = math.sin(math.radians(self.cfg.termination_roll_deg))
        pitch_lim = math.sin(math.radians(self.cfg.termination_pitch_deg))
        orientation_term = (pgb[:, 0].abs() > pitch_lim) | (pgb[:, 1].abs() > roll_lim)
        base_low_term = self.base_height < self.cfg.terminate_base_height_low
        nan_term = (
            ~torch.isfinite(self.robot.data.joint_pos).all(dim=-1)
            | ~torch.isfinite(self.robot.data.root_lin_vel_b).all(dim=-1)
            | ~torch.isfinite(self.robot.data.root_ang_vel_b).all(dim=-1)
        )
        terminated = base_contact | orientation_term | base_low_term | nan_term
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    # ------------------------------------------------------------------
    # rewards（与 legged_gym 版语义一致）
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        pgb = self.robot.data.projected_gravity_b
        applied_torque = self.robot.data.applied_torque
        joint_vel = self.robot.data.joint_vel

        terms = {}
        terms["alive"] = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        terms["termination"] = self.reset_terminated.float()
        terms["flat_orientation_x_exp"] = torch.exp(
            -torch.square(pgb[:, 1]) / self.cfg.orientation_x_exp_sigma
        )
        terms["flat_orientation_y_exp"] = torch.exp(
            -torch.square(pgb[:, 0]) / self.cfg.orientation_y_exp_sigma
        )
        forces = self.wheel_contact_forces
        terms["four_wheel_contact"] = torch.clamp(
            forces / self.cfg.desired_contact_force_threshold, 0.0, 1.0
        ).mean(dim=-1)
        height_err = self.base_height - self.height_cmd
        terms["track_height_exp"] = torch.exp(
            -torch.square(torch.clamp(height_err, -0.15, 0.15)) / self.cfg.height_track_sigma
        )
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
        terms["ang_vel_xy"] = torch.sum(
            torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=-1
        )
        terms["lin_vel_z"] = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        undesired = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self._legs_contact_idx, :], dim=-1
        )
        terms["undesired_contact"] = torch.any(
            undesired > self.cfg.undesired_contact_force_threshold, dim=-1
        ).float()

        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
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

        # 关节：leg/ws 恢复默认位姿（q=0），轮子随机角（continuous 无害）
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._wheels_idx] = torch.rand(n, len(self._wheels_idx), device=device) * 4 * math.pi - 2 * math.pi
        joint_vel = torch.zeros(n, self._num_joints, device=device)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 根状态：地形 origin + 名义高度 + 缓冲；yaw 随机；速度归零（calm start）
        root_pos = self.scene.env_origins[env_ids].clone()
        root_pos[:, 2] += 0.132 + 0.005
        yaw = torch.rand(n, device=device) * 2 * math.pi - math.pi
        root_quat = torch.zeros(n, 4, device=device)
        root_quat[:, 0] = torch.cos(yaw / 2)
        root_quat[:, 3] = torch.sin(yaw / 2)
        root_vel = torch.zeros(n, 3, device=device)
        root_ang_vel = torch.zeros(n, 3, device=device)
        self.robot.write_root_pose_to_sim(root_pos, root_quat, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, root_ang_vel, env_ids=env_ids)

        # buffers
        self.height_cmd[env_ids] = torch.rand(n, device=device) * (
            self.cfg.height_range[1] - self.cfg.height_range[0]
        ) + self.cfg.height_range[0]
        self.last_actions[env_ids] = 0.0
        self.leg_target[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self._legs_idx]
        self._prev_joint_vel[env_ids] = 0.0

        # 日志
        self.extras["log"] = {}
        for name, sums in self.episode_sums.items():
            self.extras["log"][f"episode/{name}"] = sums[env_ids].mean().item()
            sums[env_ids] = 0.0
