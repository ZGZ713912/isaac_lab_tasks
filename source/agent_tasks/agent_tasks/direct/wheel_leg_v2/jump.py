# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
#
# Wheel_leg_V2 跳跃控制器（轮腿原地/行进跳跃）。
#
# 语义照搬 wheelbipe 的 jump_takeoff / airborne 状态机（见
# source/agent_tasks/agent_tasks/direct/wheelbipe/state_machines/）：
#   1. 只在"稳定"时随机/受令触发；弹道参考（峰值高度→离地速度→蹬伸时间）解析求解；
#   2. 相位 IDLE→PUSH→TUCK→IDLE，PUSH 内按参考竖直速度给奖励，TUCK 内奖励滞空；
#   3. 早期用"缺失速度"辅助力（对 base 施加世界 +Z 力）帮助 bootstrap，按训练进度衰减；
#   4. 落地检测（离地→触地）给向下速度惩罚。
#
# 本模块是纯 Torch，不依赖 Isaac。
# =============================================================================

from __future__ import annotations

import math

import torch

PHASE_IDLE = 0
PHASE_PUSH = 1
PHASE_TUCK = 2


class JumpController:
    """按 policy step 推进的跳跃相位/弹道参考/辅助力。"""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        peak_height_range: tuple[float, float] = (0.35, 0.45),
        push_start_height_m: float = 0.23,
        release_height_m: float = 0.29,
        fixed_t_push_s: float = 0.12,
        exit_time_scale: float = 1.3,
        min_duration_s: float = 0.30,
        cooldown_s: float = 1.5,
        trigger_rate_per_s: float = 0.5,
        trigger_min_episode_time_s: float = 3.0,
        gravity: float = 9.81,
    ):
        self.num_envs = int(num_envs)
        self.device = device
        self.peak_height_range = (float(peak_height_range[0]), float(peak_height_range[1]))
        self.push_start_height_m = float(push_start_height_m)
        self.release_height_m = float(release_height_m)
        self.fixed_t_push_s = float(fixed_t_push_s)
        self.exit_time_scale = float(exit_time_scale)
        self.min_duration_s = float(min_duration_s)
        self.cooldown_s = float(cooldown_s)
        self.trigger_rate_per_s = float(trigger_rate_per_s)
        self.trigger_min_episode_time_s = float(trigger_min_episode_time_s)
        self.gravity = float(gravity)

        z = lambda: torch.zeros(self.num_envs, device=device)  # noqa: E731
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.phase_time = z()
        self.cooldown = z()
        self.request = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.target_peak = z()
        self.ref_v_rel = z()
        self.ref_t_push = torch.full((self.num_envs,), self.fixed_t_push_s, device=device)
        self.ref_duration = torch.full((self.num_envs,), self.min_duration_s, device=device)
        self.ref_vel_z = z()
        self.push_max_vel_z = z()
        self.max_height = z()
        self.air_time = z()
        self.assist_active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.assist_force_z = z()
        self.exit_event = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.landing_event = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.landing_down_vel = z()
        self.prev_airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.phase[env_ids] = PHASE_IDLE
        self.phase_time[env_ids] = 0.0
        self.cooldown[env_ids] = 0.0
        self.request[env_ids] = False
        self.ref_vel_z[env_ids] = 0.0
        self.push_max_vel_z[env_ids] = 0.0
        self.max_height[env_ids] = 0.0
        self.air_time[env_ids] = 0.0
        self.assist_active[env_ids] = False
        self.assist_force_z[env_ids] = 0.0
        self.exit_event[env_ids] = False
        self.landing_event[env_ids] = False
        self.landing_down_vel[env_ids] = 0.0
        self.prev_airborne[env_ids] = False

    def request_jump(self, env_ids: torch.Tensor) -> None:
        self.request[env_ids] = True

    # ------------------------------------------------------------------
    def step(
        self,
        dt: float,
        base_height: torch.Tensor,
        vel_z: torch.Tensor,
        episode_time: torch.Tensor,
        airborne: torch.Tensor,
        assist_probability: float,
        force_z: float,
        missing_vel_gain: float,
        max_force_z: float,
    ) -> None:
        """推进一个 policy step。airborne: 两轮均离地的布尔量。"""
        idle = self.phase == PHASE_IDLE
        stable = episode_time >= self.trigger_min_episode_time_s
        cooldown_ok = self.cooldown <= 0.0
        # 触发：外部 request 或按速率随机
        random = torch.rand(self.num_envs, device=self.device) < self.trigger_rate_per_s * dt
        enter = idle & (self.request | random) & stable & cooldown_ok

        low, high = self.peak_height_range
        peak = low + (high - low) * torch.rand(self.num_envs, device=self.device)
        t_flight = torch.sqrt(2.0 * (peak - self.release_height_m).clamp_min(0.0) / self.gravity)
        v_rel = self.gravity * t_flight
        push_dist = max(self.release_height_m - self.push_start_height_m, 0.0)
        t_push = torch.where(
            v_rel > 1e-3,
            torch.full_like(v_rel, 2.0 * push_dist) / v_rel.clamp_min(1e-6),
            torch.full_like(v_rel, self.fixed_t_push_s),
        )
        duration = (t_push + self.exit_time_scale * t_flight).clamp_min(self.min_duration_s)

        self.target_peak = torch.where(enter, peak, self.target_peak)
        self.ref_v_rel = torch.where(enter, v_rel, self.ref_v_rel)
        self.ref_t_push = torch.where(enter, t_push, self.ref_t_push)
        self.ref_duration = torch.where(enter, duration, self.ref_duration)
        self.phase = torch.where(enter, torch.full_like(self.phase, PHASE_PUSH), self.phase)
        self.phase_time = torch.where(enter, torch.zeros_like(self.phase_time), self.phase_time)
        self.push_max_vel_z = torch.where(enter, torch.zeros_like(self.push_max_vel_z), self.push_max_vel_z)
        self.max_height = torch.where(enter, base_height, self.max_height)
        self.air_time = torch.where(enter, torch.zeros_like(self.air_time), self.air_time)
        self.request = torch.where(enter, torch.zeros_like(self.request), self.request)
        self.cooldown = torch.where(enter, torch.full_like(self.cooldown, self.cooldown_s), self.cooldown)
        assist_draw = torch.rand(self.num_envs, device=self.device) < assist_probability
        self.assist_active = torch.where(enter & assist_draw, torch.ones_like(self.assist_active), self.assist_active)

        # —— 推进活动相位 ——
        active = self.phase != PHASE_IDLE
        phase_time = self.phase_time + dt * active.to(self.phase_time.dtype)
        t_push = self.ref_t_push
        v_rel = self.ref_v_rel
        push_accel = v_rel / t_push.clamp_min(1e-6)
        in_push = self.phase == PHASE_PUSH
        ref_vel_z = torch.where(
            phase_time < t_push,
            push_accel * phase_time,
            v_rel - self.gravity * (phase_time - t_push),
        )
        ref_vel_z = ref_vel_z.clamp_min(0.0)
        self.push_max_vel_z = torch.where(active, torch.maximum(self.push_max_vel_z, vel_z), self.push_max_vel_z)
        self.max_height = torch.where(active, torch.maximum(self.max_height, base_height), self.max_height)
        self.air_time = torch.where(active & airborne, self.air_time + dt, self.air_time)

        to_tuck = in_push & (phase_time >= t_push)
        self.phase = torch.where(to_tuck, torch.full_like(self.phase, PHASE_TUCK), self.phase)
        self.ref_vel_z = ref_vel_z

        done = active & (phase_time >= self.ref_duration)
        self.exit_event = done
        self.phase = torch.where(done, torch.full_like(self.phase, PHASE_IDLE), self.phase)
        self.phase_time = torch.where(done, torch.zeros_like(phase_time), phase_time)
        self.cooldown = torch.where(done, torch.full_like(self.cooldown, self.cooldown_s),
                                    self.cooldown - dt)

        # 辅助力只作用于 PUSH 相位
        self.assist_active = self.assist_active & (self.phase == PHASE_PUSH)
        missing = (v_rel - vel_z).clamp_min(0.0)
        raw_force = force_z + missing_vel_gain * missing
        raw_force = raw_force.clamp_max(max_force_z)
        self.assist_force_z = torch.where(
            self.assist_active, raw_force, torch.zeros_like(raw_force))

        # 落地检测：上一步离地、这一步触地
        self.landing_event = self.prev_airborne & (~airborne)
        self.landing_down_vel = torch.where(
            self.landing_event, (-vel_z).clamp_min(0.0), self.landing_down_vel)
        self.prev_airborne = airborne

    # ------------------------------------------------------------------
    def reward_additions(
        self,
        vel_z: torch.Tensor,
        sigma_push: float = 0.5,
        sigma_peak: float = 0.05,
        push_max_vel_cap: float = 50.0,
    ) -> dict[str, torch.Tensor]:
        """返回未加权（未乘 dt）的跳跃奖励项；env 负责乘权重 × policy_dt。"""
        active = self.phase != PHASE_IDLE
        mask_push = (self.phase == PHASE_PUSH).to(vel_z.dtype)
        mask_active = active.to(vel_z.dtype)
        push_track = mask_push * torch.exp(-((vel_z - self.ref_vel_z) / max(sigma_push, 1e-6)).square())
        push_max = mask_active * self.push_max_vel_z.clamp(0.0, push_max_vel_cap)
        peak_track = self.exit_event.to(vel_z.dtype) * torch.exp(
            -((self.max_height - self.target_peak) / max(sigma_peak, 1e-6)).square())
        return {
            "jump_push_track": push_track,
            "jump_push_max_vel": push_max,
            "jump_peak_track": peak_track,
            "jump_air_time": mask_active * self.air_time,
            "airborne_landing_down_vel": self.landing_event.to(vel_z.dtype) * self.landing_down_vel,
        }
