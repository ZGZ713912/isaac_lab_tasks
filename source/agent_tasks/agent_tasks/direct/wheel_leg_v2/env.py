# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V2 闭链轮腿任务的 DirectRLEnv。
#
# 架构移植自 V40 训练仓 wheeled-biped-rl-train/src/wheeled_tasks/direct/v40_serial/env.py：
#   6 个等效输出关节的 joint-space PD + 轮 torque-speed 曲线，
#   25D 观测 ×5 历史、29D critic、奖励核与 V40 合同一致。
# 适配点（V2 闭链）：
#   - 资产是已 authored 的 Wheel_leg_V2.usd，18 个树关节中只驱动 6 个，
#     其余 12 个四杆/气弹簧关节为被动，仅由 spherical/prismatic 闭合约束跟随；
#   - 额外做闭链误差监控（对标 validate_wheel_leg_v2_closed.py），超限即安全 reset；
#   - 气弹簧第一版不加力（纯约束）。
# =============================================================================

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from agent_world.assets.wheel_leg_V2 import WheelLegV2_CFG

from .contract import asset_paths, is_round2, load_constraints, load_contract
from .core import (
    HistoryStack,
    NoisyHistoryStack,
    SustainedFailure,
    build_critic,
    build_observation,
    compute_reward_terms,
    compute_torques,
    decode_targets,
)
from .env_cfg import WheelLegV2EnvCfg

# USD / URDF 里的 19 个刚体（base + 左右各 9）
BODY_NAMES = [
    "base_link",
    "L_link1", "L_link2", "L_link3",
    "LL_link1", "LL_link2", "LL_link3", "LL_link4",
    "LLL_link1", "LLL_link2",
    "R_link1", "R_link2", "R_link3",
    "RR_link1", "RR_link2", "RR_link3", "RR_link4",
    "RRR_link1", "RRR_link2",
]
WHEEL_BODY_NAMES = ["L_link3", "R_link3"]
NON_WHEEL_BODY_NAMES = [name for name in BODY_NAMES if name not in WHEEL_BODY_NAMES]
# 需要监控的闭合点涉及的全部刚体
CLOSURE_BODY_NAMES = ["L_link1", "L_link2", "LL_link3", "LL_link4", "R_link1", "R_link2", "RR_link3", "RR_link4"]


class WheelLegV2Env(DirectRLEnv):
    cfg: WheelLegV2EnvCfg

    def __init__(self, cfg: WheelLegV2EnvCfg, render_mode: str | None = None, **kwargs):
        # —— 合同驱动：在 super().__init__ 构造场景之前填好 dims/时钟/资产 ——
        self.contract = load_contract(cfg.contract_path)
        self.constraints = load_constraints(self.contract)
        paths = asset_paths(self.contract)
        if not paths["usd"].is_file():
            raise FileNotFoundError(f"Wheel_leg_V2 USD not found: {paths['usd']}")
        if cfg.stage not in self.contract["commands"]["stages"]:
            raise ValueError(f"unknown Wheel_leg_V2 stage: {cfg.stage}")
        timing = self.contract["timing"]
        cfg.sim.dt = timing["physics_dt"]
        cfg.decimation = timing["decimation"]
        cfg.sim.render_interval = cfg.decimation
        cfg.episode_length_s = self.contract["termination"]["episode_seconds"]
        cfg.observation_space = self.contract["observations"]["actor_dim"]
        cfg.state_space = self.contract["observations"]["critic_dim"]
        cfg.action_space = self.contract["actions"]["dimension"]
        cfg.is_finite_horizon = False

        self.joint_names = list(self.contract["joints"]["action_order"])
        nominal = self.contract["joints"]["nominal_positions"]
        base_height = self.contract["asset"]["nominal_base_height"]
        # 直接复用资产 cfg，只改 prim_path 与初始根高
        robot_cfg = WheelLegV2_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()
        robot_cfg.init_state.pos = (0.0, 0.0, base_height)
        robot_cfg.init_state.joint_pos = {".*": 0.0}
        robot_cfg.init_state.joint_vel = {".*": 0.0}
        cfg.robot_cfg = robot_cfg

        body_expression = "(" + "|".join(re.escape(name) for name in BODY_NAMES) + ")"
        cfg.contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/" + body_expression,
            update_period=0.0, history_length=cfg.decimation, debug_vis=False,
        )

        super().__init__(cfg, render_mode, **kwargs)

        # —— 名称解析（按名字，不依赖导入顺序）——
        self._joint_ids = self._named_indices(self.robot.joint_names, self.joint_names, "joint")
        self._wheel_body_ids = self._named_indices(self.contact_sensor.body_names, WHEEL_BODY_NAMES, "wheel body")
        self._non_wheel_body_ids = self._named_indices(
            self.contact_sensor.body_names, NON_WHEEL_BODY_NAMES, "non-wheel body")
        self._closure_body_ids = self._named_indices(self.robot.body_names, CLOSURE_BODY_NAMES, "closure body")
        self._body_ids = self._named_indices(self.robot.body_names, BODY_NAMES, "robot body")

        joints = self.contract["joints"]
        self._leg_ids = torch.tensor(joints["leg_indices"], dtype=torch.long, device=self.device)
        self._wheel_ids = torch.tensor(joints["wheel_indices"], dtype=torch.long, device=self.device)
        self._nominal = torch.tensor(nominal, device=self.device)
        hard = joints.get("knee_hard_limits", {})
        self._knee_ids = torch.tensor(
            [i for i in joints["knee_indices"] if joints["action_order"][i] in hard],
            dtype=torch.long, device=self.device,
        )
        if self._knee_ids.numel():
            self._knee_limits = torch.tensor(
                [hard[joints["action_order"][int(i)]] for i in self._knee_ids], device=self.device)
        else:
            self._knee_limits = torch.zeros((0, 2), device=self.device)

        # —— 运行时状态 ——
        self.actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.leg_targets = self._nominal[self._leg_ids].repeat(self.num_envs, 1)
        self.wheel_targets = torch.zeros((self.num_envs, 2), device=self.device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._command_period_ticks = max(1, math.ceil(
            self.contract["commands"]["resample_seconds"] / self.contract["timing"]["policy_dt"]))
        self._command_ticks_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._commands_due = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._invalid_actions = torch.zeros_like(self._commands_due)
        self._finite_state = torch.ones_like(self._commands_due)
        self._last_reward_tick = -1
        self._episode_sums: dict[str, torch.Tensor] = {}

        history_type = NoisyHistoryStack if is_round2(self.contract) else HistoryStack
        self.history = history_type(
            self.num_envs, self.device,
            length=self.contract["observations"]["history_length"],
            dim=self.contract["observations"]["single_dim"],
        )
        if is_round2(self.contract):
            self._sustained_failure = SustainedFailure(self.num_envs, self.device)
        self._evaluation_command = cfg.evaluation_command
        self._sample_commands(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    # ------------------------------------------------------------------ setup
    def _named_indices(self, actual: list[str], requested: list[str], kind: str) -> torch.Tensor:
        if any(actual.count(name) != 1 for name in requested):
            raise RuntimeError(f"missing/ambiguous {kind} names: expected {requested}, received {actual}")
        return torch.tensor([actual.index(name) for name in requested], device=self.device, dtype=torch.long)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact"] = self.contact_sensor
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=True)
        # 独立克隆需要显式跨环境过滤；USD 内已自带自碰撞过滤。
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ state
    def _joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.robot.data.joint_pos[:, self._joint_ids],
                self.robot.data.joint_vel[:, self._joint_ids])

    def _base_height(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    def _contact_magnitudes(self) -> torch.Tensor:
        history = self.contact_sensor.data.net_forces_w_history
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=1)

    def _closure_errors(self) -> dict[str, torch.Tensor]:
        """每个闭合点两杆世界点距离（米），对标 validate_wheel_leg_v2_closed.py。"""
        from isaaclab.utils.math import quat_apply

        body_pos = self.robot.data.body_pos_w
        body_quat = self.robot.data.body_quat_w
        errors = {}
        for closure in self.constraints["closures"]:
            i0 = self._closure_body_ids[CLOSURE_BODY_NAMES.index(closure["body0"])]
            i1 = self._closure_body_ids[CLOSURE_BODY_NAMES.index(closure["body1"])]
            p0 = torch.tensor(closure["local_pos0_m"], dtype=body_pos.dtype, device=self.device)
            p1 = torch.tensor(closure["local_pos1_m"], dtype=body_pos.dtype, device=self.device)
            w0 = body_pos[:, i0] + quat_apply(body_quat[:, i0], p0.expand_as(body_pos[:, i0]))
            w1 = body_pos[:, i1] + quat_apply(body_quat[:, i1], p1.expand_as(body_pos[:, i1]))
            errors[closure["name"]] = torch.linalg.vector_norm(w0 - w1, dim=-1)
        return errors

    # ------------------------------------------------------------- action
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 6):
            raise ValueError("expected N x 6 actions")
        self.previous_actions.copy_(self.actions)
        self._invalid_actions = ~torch.isfinite(actions).all(dim=-1)
        safe_actions = torch.where(self._invalid_actions[:, None], torch.zeros_like(actions), actions)
        joint_pos, _ = self._joint_state()
        self.leg_targets, self.wheel_targets, clipped = decode_targets(safe_actions, joint_pos, self.contract)
        self.actions.copy_(clipped)

    def _apply_action(self) -> None:
        # DirectRLEnv 每个物理子步都调用；反馈不能冻结在策略频率。
        joint_pos, joint_vel = self._joint_state()
        finite = (torch.isfinite(joint_pos).all(-1) & torch.isfinite(joint_vel).all(-1)
                  & ~self._invalid_actions)
        self._invalid_actions |= ~finite
        self.torques.zero_()
        if finite.any():
            self.torques[finite] = compute_torques(
                joint_pos[finite], joint_vel[finite], self.leg_targets[finite],
                self.wheel_targets[finite], self.contract,
            )
        bad = ~torch.isfinite(self.torques).all(-1)
        self._invalid_actions |= bad
        self.torques[bad] = 0.0
        # 只对 6 个驱动关节下发 effort；被动关节保持 0。
        self.robot.set_joint_effort_target(self.torques, joint_ids=self._joint_ids)

    # ------------------------------------------------------------- dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos, joint_vel = self._joint_state()
        body = self.robot.data
        contact = self._contact_magnitudes()
        self._finite_state = (
            torch.isfinite(body.root_state_w).all(-1)
            & torch.isfinite(joint_pos).all(-1) & torch.isfinite(joint_vel).all(-1)
            & torch.isfinite(contact).all(-1) & torch.isfinite(self.torques).all(-1)
            & ~self._invalid_actions
        )
        limits = self.contract["termination"]
        non_wheel_contact = (contact[:, self._non_wheel_body_ids] > limits["contact_force_threshold"]).any(-1)
        if self._knee_ids.numel():
            knee_q = joint_pos[:, self._knee_ids]
            tol = limits.get("knee_limit_tolerance", 0.0)
            knee_out = ((knee_q < self._knee_limits[:, 0] - tol)
                        | (knee_q > self._knee_limits[:, 1] + tol)).any(-1)
        else:
            knee_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        tilt = -body.projected_gravity_b[:, 2] < math.cos(math.radians(limits["max_tilt_deg"]))
        low = self._base_height() < limits["min_base_height"]

        closure = self._closure_errors()
        closure_stack = torch.stack(tuple(closure.values()), dim=-1)
        closure_max = closure_stack.amax(dim=-1)
        tol = float(self.cfg.closure_error_tolerance)
        closure_bad = closure_max > tol if tol > 0.0 else torch.zeros_like(tilt)

        diagnostics = {"non_wheel_contact": non_wheel_contact, "knee_limit": knee_out,
                       "tilt": tilt, "low_height": low, "closure": closure_bad}
        reasons = {"nonfinite": ~self._finite_state, **diagnostics}
        if is_round2(self.contract):
            self._finite_state &= torch.isfinite(body.projected_gravity_b).all(-1)
            sustained = self._sustained_failure.update(
                body.projected_gravity_b[:, 2], int(self.common_step_counter), self.contract)
            reasons = {name: torch.zeros_like(tilt) for name in diagnostics}
            reasons.update(nonfinite=~self._finite_state, tilt=sustained)
            terminated = ~self._finite_state | sustained | closure_bad
        else:
            terminated = ~self._finite_state | non_wheel_contact | knee_out | tilt | low | closure_bad

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        log = self.extras.setdefault("log", {})
        for name, flag in {**reasons, "timeout": time_out}.items():
            log[f"Termination/{name}"] = flag.float().mean()
        log["Geometry/base_height_m"] = torch.nan_to_num(self._base_height(), nan=0.0).mean()
        log["Geometry/closure_max_mm"] = torch.nan_to_num(closure_max * 1000.0, nan=0.0).mean()
        for name, value in closure.items():
            log[f"Closure/{name}_mm"] = torch.nan_to_num(value * 1000.0, nan=0.0).mean()
        for side, body_id in zip(("left", "right"), self._wheel_body_ids):
            force = torch.nan_to_num(contact[:, body_id], nan=0.0, posinf=0.0, neginf=0.0)
            log[f"Contact/{side}_wheel_force_n"] = force.mean()
        return terminated, time_out

    # ------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        joint_pos, _ = self._joint_state()
        data = self.robot.data
        valid = self._finite_state
        terms = compute_reward_terms(
            data.root_lin_vel_b[valid], data.root_ang_vel_b[valid], data.projected_gravity_b[valid],
            self._base_height()[valid], self.commands[valid], self.actions[valid],
            self.previous_actions[valid], self.torques[valid], joint_pos[valid], self.contract,
        )
        total = torch.zeros(self.num_envs, device=self.device)
        log = self.extras.setdefault("log", {})
        for name, step_reward in terms.items():
            if not torch.isfinite(step_reward).all():
                raise RuntimeError(f"non-finite Wheel_leg_V2 reward term: {name}")
            value = torch.zeros_like(total)
            value[valid] = step_reward
            total += value
            log[f"Reward/{name}"] = value.mean()
            self._episode_sums.setdefault(name, torch.zeros_like(total)).add_(value)
        terminal = self.reset_terminated.float() * self.contract["rewards"]["termination_penalty"]
        total += terminal
        log["Reward/termination"] = terminal.mean()
        self._episode_sums.setdefault("termination", torch.zeros_like(total)).add_(terminal)
        for name, value in {
            "vx_m_s": data.root_lin_vel_b[:, 0], "wz_rad_s": data.root_ang_vel_b[:, 2],
            "height_m": self._base_height(),
            "vx_abs_error": (data.root_lin_vel_b[:, 0] - self.commands[:, 0]).abs(),
            "wz_abs_error": (data.root_ang_vel_b[:, 2] - self.commands[:, 1]).abs(),
            "height_abs_error": (self._base_height() - self.commands[:, 2]).abs(),
        }.items():
            log[f"Tracking/{name}"] = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).mean()
        # 命令重采样计时：奖励与跟踪使用产生本动作的旧命令，采样推迟到 _get_observations。
        tick = int(self.common_step_counter)
        if tick != self._last_reward_tick:
            self._command_ticks_left -= 1
            self._commands_due |= self._command_ticks_left <= 0
            self._last_reward_tick = tick
        return total

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._evaluation_command is not None:
            self.commands[env_ids] = torch.tensor(
                self._evaluation_command, device=self.device, dtype=self.commands.dtype)
            self._command_ticks_left[env_ids] = self._command_period_ticks
            self._commands_due[env_ids] = False
            return
        stage = self.contract["commands"]["stages"][self.cfg.stage]
        for column, key in enumerate(("vx", "wz", "height")):
            low, high = stage[key]
            self.commands[env_ids, column] = low + (high - low) * torch.rand(len(env_ids), device=self.device)
        probability = stage.get("standing_probability", 0.0) if is_round2(self.contract) else 0.0
        if probability > 0.0:
            standing = torch.rand(len(env_ids), device=self.device) < probability
            self.commands[env_ids[standing], :2] = 0.0
        self._command_ticks_left[env_ids] = self._command_period_ticks
        self._commands_due[env_ids] = False

    # ------------------------------------------------------------- observations
    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._sample_commands(self._commands_due.nonzero(as_tuple=False).flatten())
        joint_pos, joint_vel = self._joint_state()
        data = self.robot.data
        obs25 = build_observation(
            data.root_ang_vel_b, data.projected_gravity_b, self.commands,
            joint_pos, joint_vel, self.actions, self.contract,
        )
        if is_round2(self.contract):
            policy = self.history.update_actor(
                obs25, tick=int(self.common_step_counter), contract=self.contract, enabled=True)
        else:
            policy = self.history.update(obs25, tick=int(self.common_step_counter))
        critic = build_critic(obs25, data.root_lin_vel_b, self._base_height())
        return {"policy": policy, "critic": critic}

    # ------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        if self._episode_sums:
            log = self.extras.setdefault("log", {})
            duration = (self.episode_length_buf[env_ids].float() * self.step_dt).clamp_min(self.step_dt)
            for name, sums in self._episode_sums.items():
                log[f"Episode_Reward/{name}_per_second"] = (sums[env_ids] / duration).mean()
                sums[env_ids] = 0.0
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = self._nominal
        joint_vel = torch.zeros_like(joint_pos)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        root[:, 2] = self.scene.env_origins[env_ids, 2] + self.contract["asset"]["nominal_base_height"]
        root[:, 7:] = 0.0  # 保留默认根四元数
        if is_round2(self.contract):
            low, high = self.contract["reset"]["root_velocity_range"]
            root[:, 7:] = low + (high - low) * torch.rand_like(root[:, 7:])
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.leg_targets[env_ids] = self._nominal[self._leg_ids]
        self.wheel_targets[env_ids] = 0.0
        self.commands[env_ids] = 0.0
        self._command_ticks_left[env_ids] = 0
        self._commands_due[env_ids] = False
        self._invalid_actions[env_ids] = False
        self._finite_state[env_ids] = True
        self.history.reset(env_ids)
        if is_round2(self.contract):
            self._sustained_failure.reset(env_ids)
        self._sample_commands(env_ids)
        self.robot.set_joint_effort_target(self.torques[env_ids], joint_ids=self._joint_ids, env_ids=env_ids)
