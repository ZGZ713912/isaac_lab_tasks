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
#   6 个真实电机树关节的 joint-space PD + 轮 torque-speed 曲线，
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

from agent_world.actuators import WheelLegV2GasSpringModel
from agent_world.assets.wheel_leg_V2 import (
    LEGS_ACT_JOINT_NAMES,
    PASSIVE_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    WheelLegV2_CFG,
)

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
from .jump import PHASE_IDLE, PHASE_PUSH, PHASE_TUCK, JumpController
from .slope import (
    build_periodic_slope_angle_table,
    periodic_slope_gradient_torch,
    periodic_slope_height_torch,
)

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
        configured_drive_names = set(LEGS_ACT_JOINT_NAMES + WHEEL_JOINT_NAMES)
        contract_drive_names = set(self.joint_names)
        if configured_drive_names != contract_drive_names:
            raise ValueError(
                "WheelLegV2 asset/contract drive-joint mismatch: "
                f"asset={sorted(configured_drive_names)}, contract={sorted(contract_drive_names)}"
            )
        if configured_drive_names & set(PASSIVE_JOINT_NAMES):
            raise ValueError("WheelLegV2 drive and passive joint groups overlap")
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
        # 轮体在 robot 刚体表中的下标（用于轮心离地高度 / 轮底滑移计算）
        self._wheel_robot_body_ids = self._named_indices(
            self.robot.body_names, WHEEL_BODY_NAMES, "robot wheel body")

        # —— 气弹簧：loop prismatic 约束被 excludeFromArticulation，无法下发 effort，
        #    改为按 BKB 力曲线对两端刚体施加轴向力（几何来自 constraints.json）——
        self._gas_spring_enabled = bool(getattr(self.cfg, "gas_spring_enabled", False))
        self._gas_spring_model = None
        self._gas_spring_prev_length = None
        self._gas_spring_last_force = None
        self._gas_spring_body_ids = None
        self._gas_spring_body_pairs = None
        self._gas_spring_anchor_local = None
        self._gas_spring_nominal_length = None
        if self._gas_spring_enabled:
            self._setup_gas_springs()

        # —— 周期坡面地形识别（解析求高/求梯度，与地形网格同源）——
        self._setup_terrain_profile()

        # —— 跳跃控制器（仅 stage == "jump"）——
        self._jump_enabled = bool(getattr(self.cfg, "jump_enabled", False)) and self.cfg.stage == "jump"
        self.jump = None
        self._jump_base_body_ids = None
        if self._jump_enabled:
            self.jump = JumpController(
                self.num_envs, self.device,
                peak_height_range=tuple(self.cfg.jump_peak_height_range),
                push_start_height_m=float(self.cfg.jump_push_start_height_m),
                release_height_m=float(self.cfg.jump_release_height_m),
                cooldown_s=float(self.cfg.jump_cooldown_s),
                trigger_rate_per_s=float(self.cfg.jump_trigger_rate_per_s),
                trigger_min_episode_time_s=float(self.cfg.jump_min_episode_time_s),
            )
            self._jump_base_body_ids = self._body_ids[:1]

        joints = self.contract["joints"]
        self._leg_ids = torch.tensor(joints["leg_indices"], dtype=torch.long, device=self.device)
        self._wheel_ids = torch.tensor(joints["wheel_indices"], dtype=torch.long, device=self.device)
        self._hip_ids = torch.tensor(joints["hip_indices"], dtype=torch.long, device=self.device)
        self._knee_group_ids = torch.tensor(joints["knee_indices"], dtype=torch.long, device=self.device)
        # 每个驱动关节的力矩上限，用于利用率统计（与 core.compute_reward_terms 的 caps 一致）
        self._torque_cap = torch.tensor(
            [self.contract["actuators"]["wheel" if i in joints["wheel_indices"] else "leg"]["effort_limit"]
             for i in range(len(self.joint_names))],
            dtype=torch.float32, device=self.device,
        )
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
        self.previous_previous_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        # 腿关节速度 EMA，用于 leg_joint_osc 振荡惩罚
        self._leg_joint_vel_ema = torch.zeros((self.num_envs, self._leg_ids.numel()), device=self.device)
        self._leg_joint_vel_ema_alpha = 0.1
        # 力矩统计窗口：累积一个 policy step 内全部物理子步的 |τ|（decimation 次）
        self._torque_abs_sum = torch.zeros_like(self.actions)
        self._torque_abs_max = torch.zeros_like(self.actions)
        self._torque_samples = 0
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

    def _setup_gas_springs(self) -> None:
        spring_defs = list(self.constraints.get("gas_springs", []))
        if not spring_defs:
            self._gas_spring_enabled = False
            return
        self._gas_spring_model = WheelLegV2GasSpringModel(
            force_at_min_n=float(self.cfg.gas_spring_force_at_min_n),
            force_at_max_n=float(self.cfg.gas_spring_force_at_max_n),
            damping_n_s_per_m=float(self.cfg.gas_spring_damping_n_s_per_m),
        )
        spring_body_names: list[str] = []
        anchors: list[list[float]] = []
        nominal_lengths: list[float] = []
        for spring in spring_defs:
            spring_body_names.extend((spring["body0"], spring["body1"]))
            anchors.append(list(spring["local_pos0_m"]))
            anchors.append(list(spring["local_pos1_m"]))
            nominal_lengths.append(float(spring["nominal_length_m"]))
        self._gas_spring_body_ids = self._named_indices(
            self.robot.body_names, spring_body_names, "gas spring body")
        self._gas_spring_anchor_local = torch.tensor(anchors, dtype=torch.float32, device=self.device)
        self._gas_spring_nominal_length = torch.tensor(
            nominal_lengths, dtype=torch.float32, device=self.device)
        # 每个弹簧的 body0/body1 在 _gas_spring_body_ids 中的位置
        self._gas_spring_body_pairs = torch.arange(
            len(spring_body_names), dtype=torch.long, device=self.device).reshape(-1, 2)
        self._gas_spring_prev_length = self._gas_spring_nominal_length.repeat(self.num_envs, 1)
        self._gas_spring_last_force = torch.zeros_like(self._gas_spring_prev_length)

    def _apply_gas_spring_forces(self) -> None:
        """按气弹簧力曲线对两杆施加轴向力（世界力→各刚体局部系，作用在各杆销点）。"""
        if self._gas_spring_model is None or self._gas_spring_prev_length is None:
            return
        from isaaclab.utils.math import quat_apply_inverse

        pos = self.robot.data.body_pos_w[:, self._gas_spring_body_ids]      # [N, B, 3]
        quat = self.robot.data.body_quat_w[:, self._gas_spring_body_ids]    # [N, B, 4]
        pairs = self._gas_spring_body_pairs                                 # [S, 2]
        p0, p1 = pos[:, pairs[:, 0]], pos[:, pairs[:, 1]]
        delta = p1 - p0
        length = torch.linalg.vector_norm(delta, dim=-1)                    # [N, S]
        axis = delta / length.clamp_min(1e-6).unsqueeze(-1)                 # body0 -> body1
        dt = max(float(self.physics_dt), 1e-9)
        rate = (length - self._gas_spring_prev_length) / dt
        magnitude = self._gas_spring_model.force_magnitude(length, rate)    # [N, S]
        force_local = torch.zeros_like(pos)
        for column, body_pair in enumerate(pairs.tolist()):
            i0, i1 = body_pair
            f0 = (-magnitude[:, column]).unsqueeze(-1) * axis[:, column]
            f1 = (magnitude[:, column]).unsqueeze(-1) * axis[:, column]
            force_local[:, i0] = quat_apply_inverse(quat[:, i0], f0)
            force_local[:, i1] = quat_apply_inverse(quat[:, i1], f1)
        finite = torch.isfinite(force_local).all(dim=-1).all(dim=-1)
        force_local = torch.where(finite[:, None, None], force_local, torch.zeros_like(force_local))
        anchors = self._gas_spring_anchor_local.unsqueeze(0).expand(self.num_envs, -1, -1)
        torques = torch.zeros_like(pos)
        self.robot.set_external_force_and_torque(
            forces=force_local, torques=torques, positions=anchors,
            body_ids=self._gas_spring_body_ids,
        )
        self._gas_spring_prev_length.copy_(length.detach())
        self._gas_spring_last_force = magnitude.detach()

    def _setup_terrain_profile(self) -> None:
        """识别周期坡面地形，构建坡角表与剖面坐标偏移（平地时全部关闭）。"""
        terrain = getattr(self.cfg, "terrain", None)
        gen = getattr(terrain, "terrain_generator", None) if terrain is not None else None
        sub = None
        if gen is not None and getattr(gen, "sub_terrains", None):
            sub = gen.sub_terrains.get("periodic_slope")
        self._periodic = sub is not None
        self._period_seg = 0.0
        self._slope_angle_table = None
        self._profile_x_offset = 0.0
        self._terrain_half_size = None
        if not self._periodic:
            return
        self._period_seg = float(sub.segment_length)
        size = gen.size
        num_periods = int(math.ceil(float(size[0]) / (4.0 * self._period_seg))) + 2
        self._slope_angle_table = build_periodic_slope_angle_table(
            self._period_seg, tuple(sub.angle_range), int(sub.angle_seed), num_periods, self.device
        )
        # 地形网格以世界原点为中心（[-size/2, size/2]）；剖面坐标 p = world_x + size/2。
        self._profile_x_offset = float(size[0]) * 0.5
        self._terrain_half_size = (float(size[0]) * 0.5, float(size[1]) * 0.5)

    def _ground_height_at_x(self, x: torch.Tensor) -> torch.Tensor:
        if self._periodic:
            return periodic_slope_height_torch(
                x + self._profile_x_offset, self._period_seg, self._slope_angle_table)
        return torch.zeros_like(x)

    def _terrain_normal_w(self) -> torch.Tensor:
        """脚下地面在世界系的外法向，形状 (N,3)。"""
        normal = torch.zeros(self.num_envs, 3, device=self.device)
        normal[:, 2] = 1.0
        if self._periodic:
            gradient = periodic_slope_gradient_torch(
                self.robot.data.root_pos_w[:, 0] + self._profile_x_offset,
                self._period_seg, self._slope_angle_table)
            normal[:, 0] = -gradient
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return normal

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact"] = self.contact_sensor
        terrain = getattr(self.cfg, "terrain", None)
        if terrain is not None:
            terrain.num_envs = self.scene.cfg.num_envs
            terrain.env_spacing = self.scene.cfg.env_spacing
            self.terrain = terrain.class_type(terrain)
        else:
            self.terrain = None
            spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(physics_material=self.cfg.sim.physics_material),
        )
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
        """车体原点相对脚下地面的高度（周期坡面用解析地面高度）。"""
        if self._periodic:
            return self.robot.data.root_pos_w[:, 2] - self._ground_height_at_x(
                self.robot.data.root_pos_w[:, 0])
        return self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    def _contact_magnitudes(self) -> torch.Tensor:
        history = self.contact_sensor.data.net_forces_w_history
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=1)

    def _wheel_clearance(self) -> torch.Tensor:
        """轮心相对地面高度减去 (轮半径 + 容差)：>0 表示离地（弹跳），<0 表示压在地面。"""
        rewards = self.contract["rewards"]
        radius = float(rewards.get("wheel_hop_reference_radius", 0.06))
        tolerance = float(rewards.get("wheel_hop_clearance_tolerance", 0.01))
        wheel_pos = self.robot.data.body_pos_w[:, self._wheel_robot_body_ids]
        height = wheel_pos[..., 2] - self._ground_height_at_x(wheel_pos[..., 0])
        return height - (radius + tolerance)

    def _wheel_contact_flags(self) -> torch.Tensor:
        """每轮触地标志（浮点，1=接触力超过阈值）。"""
        threshold = float(self.contract["rewards"].get("wheel_contact_force_threshold", 5.0))
        force = self._contact_magnitudes()[:, self._wheel_body_ids]
        return (force > threshold).to(torch.float32)

    def _wheel_slip(self) -> torch.Tensor:
        """轮底接触点相对地面的切向滑移速度平方（纯滚动 ≈ 0）。"""
        radius = float(self.contract["rewards"].get("wheel_slip_reference_radius", 0.06))
        lin_w = getattr(self.robot.data, "body_lin_vel_w", None)
        ang_w = getattr(self.robot.data, "body_ang_vel_w", None)
        count = self._wheel_robot_body_ids.numel()
        if lin_w is None or ang_w is None or count == 0:
            return torch.zeros((self.num_envs, count), device=self.device)
        lin = lin_w[:, self._wheel_robot_body_ids]
        ang = ang_w[:, self._wheel_robot_body_ids]
        offset = torch.zeros_like(lin)
        offset[..., 2] = -radius
        bottom_vel = lin + torch.cross(ang, offset, dim=-1)
        return bottom_vel[..., :2].square().sum(dim=-1)

    def _wheel_x_relative_to_root(self) -> torch.Tensor:
        """轮心在根部坐标系的前后位置，x>0 表示位于质心前方。"""
        from isaaclab.utils.math import quat_apply_inverse

        wheel_pos_w = self.robot.data.body_pos_w[:, self._wheel_robot_body_ids]
        root_pos_w = self.robot.data.root_pos_w[:, None, :]
        root_quat_w = self.robot.data.root_quat_w[:, None, :].expand(
            -1, wheel_pos_w.shape[1], -1)
        wheel_pos_b = quat_apply_inverse(root_quat_w, wheel_pos_w - root_pos_w)
        return wheel_pos_b[..., 0]

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
    def _jump_assist_probability(self) -> float:
        start = float(self.cfg.jump_assist_prob_start)
        end = float(self.cfg.jump_assist_prob_end)
        decay_steps = max(1, int(self.cfg.jump_assist_decay_iterations) * int(self.cfg.jump_steps_per_iteration))
        fraction = min(1.0, float(self.common_step_counter) / decay_steps)
        return start + (end - start) * fraction

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 6):
            raise ValueError("expected N x 6 actions")
        self.previous_previous_actions.copy_(self.previous_actions)
        self.previous_actions.copy_(self.actions)
        self._invalid_actions = ~torch.isfinite(actions).all(dim=-1)
        safe_actions = torch.where(self._invalid_actions[:, None], torch.zeros_like(actions), actions)
        joint_pos, _ = self._joint_state()
        self.leg_targets, self.wheel_targets, clipped = decode_targets(safe_actions, joint_pos, self.contract)
        self.actions.copy_(clipped)
        if self.jump is not None:
            airborne = (self._wheel_clearance() > 0.0).all(dim=-1)
            root_vel_z = torch.nan_to_num(
                self.robot.data.root_lin_vel_w[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
            self.jump.step(
                float(self.step_dt),
                torch.nan_to_num(self._base_height(), nan=0.25),
                root_vel_z,
                self.episode_length_buf.to(torch.float32) * self.step_dt,
                airborne,
                self._jump_assist_probability(),
                float(self.cfg.jump_assist_force_z),
                float(self.cfg.jump_assist_missing_vel_gain),
                float(self.cfg.jump_assist_max_force_z),
            )

    def _apply_jump_assist(self) -> None:
        """对 base 施加世界 +Z 辅助力（帮助 bootstrap 蹬伸）；非 PUSH 相位力为 0。"""
        if self.jump is None:
            return
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces[:, 0, 2] = torch.nan_to_num(
            self.jump.assist_force_z, nan=0.0, posinf=0.0, neginf=0.0)
        torques = torch.zeros_like(forces)
        self.robot.set_external_force_and_torque(
            forces=forces, torques=torques, body_ids=self._jump_base_body_ids, is_global=True)

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
        # 记录本子步力矩（|τ| 累加与峰值），供 _get_dones 输出到 extras["log"]
        abs_tau = self.torques.abs()
        self._torque_abs_sum += abs_tau
        torch.maximum(self._torque_abs_max, abs_tau, out=self._torque_abs_max)
        self._torque_samples += 1
        # 只对 6 个真实电机关节下发 effort；被动关节保持 0。
        self.robot.set_joint_effort_target(self.torques, joint_ids=self._joint_ids)
        # 气弹簧被动力（每物理子步重算，避免冻结在策略频率）。
        self._apply_gas_spring_forces()
        # 跳跃辅助力（仅 jump 任务；非 PUSH 相位自动置 0）。
        self._apply_jump_assist()

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
        if self._terrain_half_size is not None:
            margin = 3.0
            root_xy = self.robot.data.root_pos_w[:, :2]
            off = ((root_xy[:, 0].abs() > self._terrain_half_size[0] - margin)
                   | (root_xy[:, 1].abs() > self._terrain_half_size[1] - margin))
            time_out = time_out | off

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
        clearance = self._wheel_clearance()
        log["Geometry/wheel_clearance_max_m"] = clearance.clamp(min=0.0).amax(dim=-1).mean()
        log["Geometry/wheel_slip_max"] = self._wheel_slip().amax(dim=-1).mean()
        log["Contact/wheels_grounded_frac"] = self._wheel_contact_flags().all(dim=-1).float().mean()
        wheel_x = self._wheel_x_relative_to_root()
        log["Geometry/wheel_x_rel_root_left_m"] = wheel_x[:, 0].mean()
        log["Geometry/wheel_x_rel_root_right_m"] = wheel_x[:, 1].mean()
        log["Geometry/wheel_x_rel_root_min_m"] = wheel_x.amin(dim=-1).mean()
        pitch = torch.rad2deg(torch.asin(body.projected_gravity_b[:, 0].clamp(-1.0, 1.0)))
        roll = torch.rad2deg(torch.asin(body.projected_gravity_b[:, 1].clamp(-1.0, 1.0)))
        log["Orientation/pitch_deg"] = torch.nan_to_num(pitch, nan=0.0).mean()
        log["Orientation/roll_deg"] = torch.nan_to_num(roll, nan=0.0).mean()
        log["Orientation/pitch_rate_rad_s"] = torch.nan_to_num(body.root_ang_vel_b[:, 1], nan=0.0).mean()
        if self._gas_spring_model is not None and self._gas_spring_prev_length is not None:
            log["Spring/length_m"] = torch.nan_to_num(self._gas_spring_prev_length, nan=0.0).mean()
            log["Spring/force_n"] = torch.nan_to_num(self._gas_spring_last_force, nan=0.0).mean()
        if self.jump is not None:
            jump_active = self.jump.phase != PHASE_IDLE
            log["Jump/PUSH_frac"] = (self.jump.phase == PHASE_PUSH).float().mean()
            log["Jump/TUCK_frac"] = (self.jump.phase == PHASE_TUCK).float().mean()
            log["Jump/active_frac"] = jump_active.float().mean()
            log["Jump/assist_frac"] = self.jump.assist_active.float().mean()
            log["Jump/assist_prob"] = self._jump_assist_probability()
            log["Jump/assist_force_n"] = torch.nan_to_num(self.jump.assist_force_z, nan=0.0).mean()
            if jump_active.any():
                log["Jump/active_max_height_m"] = torch.nan_to_num(
                    self.jump.max_height[jump_active], nan=0.0).mean()
        if self._torque_samples > 0:
            abs_mean = self._torque_abs_sum / self._torque_samples
            for i, name in enumerate(self.joint_names):
                log[f"Joint/Torque_absmean/{name}"] = abs_mean[:, i].mean()
                log[f"Joint/Torque_absmax/{name}"] = self._torque_abs_max[:, i].mean()
                log[f"Joint/Torque_cmd_max/{name}"] = self._torque_abs_max[:, i].max()
                log[f"Joint/Torque_util/{name}"] = (abs_mean[:, i] / self._torque_cap[i]).mean()
            for group, ids in (("hip", self._hip_ids), ("knee", self._knee_group_ids),
                               ("wheel", self._wheel_ids)):
                log[f"Joint/Torque_absmean_{group}"] = abs_mean[:, ids].mean()
                log[f"Joint/Torque_peak_{group}"] = self._torque_abs_max[:, ids].amax(dim=1).mean()
            self._torque_abs_sum.zero_()
            self._torque_abs_max.zero_()
            self._torque_samples = 0
        return terminated, time_out

    # ------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        joint_pos, joint_vel = self._joint_state()
        data = self.robot.data
        valid = self._finite_state
        leg_vel = joint_vel[:, self._leg_ids]
        self._leg_joint_vel_ema.mul_(1.0 - self._leg_joint_vel_ema_alpha).add_(
            leg_vel, alpha=self._leg_joint_vel_ema_alpha)
        terms = compute_reward_terms(
            data.root_lin_vel_b[valid], data.root_ang_vel_b[valid], data.projected_gravity_b[valid],
            self._base_height()[valid], self.commands[valid], self.actions[valid],
            self.previous_actions[valid], self.torques[valid], joint_pos[valid], self.contract,
            wheel_clearance2=self._wheel_clearance()[valid],
            wheel_slip2=self._wheel_slip()[valid],
            wheel_contact2=self._wheel_contact_flags()[valid],
            leg_joint_vel4=leg_vel[valid],
            leg_joint_vel_ema4=self._leg_joint_vel_ema[valid],
            previous_previous_actions6=self.previous_previous_actions[valid],
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
        if self.jump is not None:
            weights = self.contract["rewards"]["weights"]
            policy_dt = self.contract["timing"]["policy_dt"]
            root_vel_z = torch.nan_to_num(data.root_lin_vel_w[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
            for name, raw in self.jump.reward_additions(root_vel_z).items():
                if not torch.isfinite(raw).all():
                    raise RuntimeError(f"non-finite Wheel_leg_V2 jump reward term: {name}")
                value = torch.where(valid, raw, torch.zeros_like(raw)) * weights.get(name, 0.0) * policy_dt
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

    def _current_iteration(self) -> int:
        """由 common_step_counter 外推训练轮次（env 不知道 runner 轮次）。

        ``iteration_steps`` 必须等于 PPO 的 num_steps_per_env；resume 时用
        ``iteration_offset`` 把已训轮次补回来，否则课程会从头开始。
        """
        steps = max(1, int(getattr(self.cfg, "iteration_steps", 1)))
        return int(self.common_step_counter) // steps + int(getattr(self.cfg, "iteration_offset", 0))

    def _sample_interval(self, spec, count: int) -> torch.Tensor:
        """采样指令区间：spec 支持 [lo,hi] 或 [[lo,hi], ...]（多区间按宽度加权）。"""
        pairs = spec if isinstance(spec[0], (list, tuple)) else [spec]
        bounds = torch.tensor(
            [[float(low), float(high)] for low, high in pairs],
            device=self.device, dtype=self.commands.dtype)
        widths = (bounds[:, 1] - bounds[:, 0]).clamp_min(0.0)
        if bounds.shape[0] == 1 or float(widths.sum()) <= 0.0:
            idx = torch.zeros(count, device=self.device, dtype=torch.long)
        else:
            idx = torch.multinomial(widths, count, replacement=True)
        low, high = bounds[idx, 0], bounds[idx, 1]
        return low + (high - low) * torch.rand(count, device=self.device)

    def _apply_command_spec(self, env_ids: torch.Tensor, spec: dict) -> None:
        count = env_ids.numel()
        for column, key in enumerate(("vx", "wz", "height")):
            self.commands[env_ids, column] = self._sample_interval(spec[key], count)

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if self._evaluation_command is not None:
            self.commands[env_ids] = torch.tensor(
                self._evaluation_command, device=self.device, dtype=self.commands.dtype)
            self._command_ticks_left[env_ids] = self._command_period_ticks
            self._commands_due[env_ids] = False
            return
        command_cfg = self.contract["commands"]
        stage = command_cfg["stages"][self.cfg.stage]
        count = env_ids.numel()
        period = torch.full(
            (count,), self._command_period_ticks,
            device=self.device, dtype=self._command_ticks_left.dtype)

        # —— 特殊模式（小陀螺自旋档）：按 rel_envs 分桶，并按训练轮次启停 ——
        modes = stage.get("special_modes", {}) or {}
        iteration = self._current_iteration()
        min_episode = float(command_cfg.get("special_mode_min_episode_seconds", 0.0))
        old_enough = (self.episode_length_buf[env_ids].float() * self.step_dt >= min_episode)
        eligible = []
        for name, mode in modes.items():
            start, end = int(mode.get("iteration_start", 0)), int(mode.get("iteration_end", -1))
            if iteration < start or (end >= 0 and iteration >= end):
                continue
            eligible.append(mode)
        mode_index = torch.full((count,), -1, device=self.device, dtype=torch.long)
        if eligible:
            u = torch.rand(count, device=self.device)
            cumulative = 0.0
            for index, mode in enumerate(eligible):
                rel = float(mode.get("rel_envs", 0.0))
                low, high = cumulative, min(cumulative + rel, 1.0)
                picked = old_enough & (mode_index < 0) & (u >= low) & (u < high)
                mode_index[picked] = index
                cumulative = high
                if cumulative >= 1.0:
                    break
            for index, mode in enumerate(eligible):
                picked = mode_index == index
                if not bool(picked.any()):
                    continue
                selected = env_ids[picked]
                spec = {key: mode.get(key, stage[key]) for key in ("vx", "wz", "height")}
                self._apply_command_spec(selected, spec)
                if mode.get("resample_seconds", 0.0) > 0.0:
                    ticks = max(1, math.ceil(
                        float(mode["resample_seconds"]) / self.contract["timing"]["policy_dt"]))
                    period[picked] = ticks

        base = mode_index < 0
        if bool(base.any()):
            base_ids = env_ids[base]
            self._apply_command_spec(base_ids, stage)
            probability = stage.get("standing_probability", 0.0) if is_round2(self.contract) else 0.0
            if probability > 0.0:
                standing = torch.rand(base_ids.numel(), device=self.device) < probability
                self.commands[base_ids[standing], :2] = 0.0

        self._command_ticks_left[env_ids] = period
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
        nominal = self.contract["asset"]["nominal_base_height"]
        if self._periodic:
            from isaaclab.utils.math import quat_from_euler_xyz

            x = root[:, 0]
            ground = self._ground_height_at_x(x)
            root[:, 2] = ground + nominal + float(self.cfg.slope_spawn_drop_m)
            gradient = periodic_slope_gradient_torch(
                x + self._profile_x_offset, self._period_seg, self._slope_angle_table)
            zeros = torch.zeros_like(x)
            yaw = (torch.rand_like(x) * 2.0 - 1.0) * math.pi
            root[:, 3:7] = quat_from_euler_xyz(zeros, torch.atan(gradient), yaw)
        else:
            root[:, 2] = self.scene.env_origins[env_ids, 2] + nominal
        root[:, 7:] = 0.0  # 保留默认根四元数（平地）
        if is_round2(self.contract):
            low, high = self.contract["reset"]["root_velocity_range"]
            root[:, 7:] = low + (high - low) * torch.rand_like(root[:, 7:])
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.previous_previous_actions[env_ids] = 0.0
        self._leg_joint_vel_ema[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self._torque_abs_sum[env_ids] = 0.0
        self._torque_abs_max[env_ids] = 0.0
        if self._gas_spring_prev_length is not None:
            self._gas_spring_prev_length[env_ids] = self._gas_spring_nominal_length
        if self.jump is not None:
            self.jump.reset(env_ids)
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
