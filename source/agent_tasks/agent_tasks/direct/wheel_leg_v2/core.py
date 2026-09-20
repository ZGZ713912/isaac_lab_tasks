# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# 纯 Torch 的 Wheel_leg_V2 观测/历史/动作/奖励数学；不依赖 Isaac。
#
# 直接移植自 V40 训练仓 wheeled-biped-rl-train/src/wheeled_tasks/v40/core.py，
# 保持同一套 25D 观测 / 6 动作 / 奖励核。差异：
#   - V2 的 knee_hard_limits 允许为空（URDF 全 continuous，机械限位待标定），
#     此时膝不做目标夹紧、soft-limit 奖励恒 0；填了限位后自动生效。
#   - V2 的名义姿态全零（SolidWorks 装配位形），由合同 nominal_positions 给出。
# =============================================================================

from __future__ import annotations

import torch

from .contract import is_round2


def _matrix(value, width, name, batch=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width or not value.is_floating_point():
        raise ValueError(f"{name} must be floating Tensor[N,{width}]")
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f"{name} batch mismatch")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return value


def _vector(value, batch, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1 or value.shape[0] != batch or not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite floating Tensor[N] or Tensor[N,1]")
    return value


def _like(values, reference):
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class HistoryStack:
    """每个 policy tick 前进一帧；reset 环境重复其首个有效帧。

    同一 tick 内重复查询返回同一历史；某 tick 内 reset 只刷新对应环境。
    """

    def __init__(self, num_envs, device, length=5, dim=25):
        if type(num_envs) is not int or num_envs < 1 or type(length) is not int or length < 1 or type(dim) is not int or dim < 1:
            raise ValueError("History dimensions must be positive integers")
        self.num_envs, self.length, self.dim = num_envs, length, dim
        self.buffer = torch.zeros((num_envs, length, dim), device=device, dtype=torch.float32)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.last_tick = None

    def reset(self, env_ids):
        raw = torch.as_tensor(env_ids, device=self.buffer.device)
        if raw.ndim == 1 and raw.numel() == 0:
            return
        if raw.dtype not in (torch.int32, torch.int64) or raw.ndim != 1:
            raise ValueError("reset expects one-dimensional integer environment ids")
        ids = raw.to(torch.long)
        if ids.numel() and ((ids < 0).any() or (ids >= self.num_envs).any()):
            raise ValueError("reset environment id out of range")
        self.buffer[ids] = 0
        self.initialized[ids] = False

    def update(self, observation, tick):
        _matrix(observation, self.dim, "history observation", self.num_envs)
        if observation.device != self.buffer.device or observation.dtype != self.buffer.dtype:
            raise ValueError("History uses float32 on its configured device")
        if type(tick) is not int or tick < 0 or self.last_tick is not None and tick < self.last_tick:
            raise ValueError("Policy tick must be a monotonic nonnegative integer")
        fresh = ~self.initialized
        if self.last_tick != tick:
            self.buffer[:, :-1] = self.buffer[:, 1:].clone()
            self.buffer[:, -1] = observation
        if fresh.any():
            self.buffer[fresh] = observation[fresh, None, :].expand(-1, self.length, -1)
            self.initialized[fresh] = True
        self.last_tick = tick
        return self.buffer.reshape(self.num_envs, self.length * self.dim).clone()


def build_observation(angular_velocity, projected_gravity, commands3,
                      joint_pos6, joint_vel6, last_actions6, contract):
    q = _matrix(joint_pos6, 6, "joint positions")
    n = q.shape[0]
    inputs = [(angular_velocity, 3, "angular velocity"), (projected_gravity, 3, "gravity"),
              (commands3, 3, "commands"), (joint_vel6, 6, "joint velocities"), (last_actions6, 6, "last actions")]
    for tensor, width, name in inputs:
        _matrix(tensor, width, name, n)
        if tensor.device != q.device or tensor.dtype != q.dtype:
            raise ValueError("Observation inputs must share dtype/device")
    j, o = contract["joints"], contract["observations"]
    error = q - _like(j["nominal_positions"], q)
    error = error.clone()
    # 只有连续髋/轮做角度 wrap；膝若为有限区不跨物理止挡，不 wrap。
    error[:, j["hip_indices"]] = wrap_angle(error[:, j["hip_indices"]])
    scales = o["scales"]
    obs = torch.cat((angular_velocity * scales["angular_velocity"], projected_gravity,
                     commands3 * _like(scales["command"], q),
                     error[:, j["leg_indices"]] * scales["joint_position"],
                     joint_vel6 * scales["joint_velocity"], last_actions6), dim=-1)
    if obs.shape != (n, 25):
        raise ValueError("Internal observation layout mismatch")
    return obs.clamp(-o["clip"], o["clip"])


def build_critic(observation25, linear_velocity_body, base_height):
    obs = _matrix(observation25, 25, "critic proprioception")
    velocity = _matrix(linear_velocity_body, 3, "critic true linear velocity", obs.shape[0])
    height = _vector(base_height, obs.shape[0], "critic height")
    return torch.cat((obs, velocity, height[:, None]), dim=-1)


class NoisyHistoryStack(HistoryStack):
    """每 tick/reset 缓存带噪 actor 帧；critic 输入仍然干净。"""

    def update_actor(self, observation, tick, contract, enabled=True):
        fresh = ~self.initialized
        rows = torch.ones_like(fresh) if self.last_tick != tick else fresh
        frame = self.buffer[:, -1].clone()
        if rows.any():
            values = observation[rows].clone()
            if enabled:
                noise = contract["observations"]["noise"]
                scales = contract["observations"]["scales"]
                amplitude = _like(
                    [noise["angular_velocity"] * scales["angular_velocity"]] * 3
                    + [noise["gravity"]] * 3 + [0.0] * 3
                    + [noise["joint_position"] * scales["joint_position"]] * 4
                    + [noise["joint_velocity"] * scales["joint_velocity"]] * 6 + [0.0] * 6,
                    values,
                )
                values += (2 * torch.rand_like(values) - 1) * amplitude
            clip = contract["observations"]["clip"]
            frame[rows] = values.clamp(-clip, clip)
        return self.update(frame, tick)


class SustainedFailure:
    """连续坏 tick 计数；独立的部分 reset。"""

    def __init__(self, num_envs, device):
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ticks = torch.full_like(self.count, -1)

    def reset(self, env_ids):
        self.count[env_ids] = 0
        self.ticks[env_ids] = -1

    def update(self, gravity_z, tick, contract):
        limits = contract["termination"]
        fresh = self.ticks != tick
        bad = gravity_z > limits["failure_gravity_z"]
        self.count[fresh] = torch.where(bad[fresh], self.count[fresh] + 1, 0)
        self.ticks[fresh] = tick
        return self.count > round(limits["failure_seconds"] / contract["timing"]["policy_dt"])


def decode_targets(actions6, joint_pos6, contract):
    q = _matrix(joint_pos6, 6, "joint positions")
    actions = _matrix(actions6, 6, "policy actions", q.shape[0])
    if actions.dtype != q.dtype or actions.device != q.device:
        raise ValueError("Action and joint state dtype/device mismatch")
    a, j = contract["actions"], contract["joints"]
    clipped = actions.clamp(-a["clip"], a["clip"])
    nominal = _like(j["nominal_positions"], q)
    desired = nominal[j["leg_indices"]] + clipped[:, j["leg_indices"]] * _like(a["leg_position_scales"], q)
    targets = desired.clone()
    for column, idx in enumerate(j["leg_indices"]):
        name = j["action_order"][idx]
        if idx in j["hip_indices"]:
            bounded = desired[:, column] if is_round2(contract) else desired[:, column].clamp(
                nominal[idx] - j["hip_soft_deviation"], nominal[idx] + j["hip_soft_deviation"]
            )
            targets[:, column] = q[:, idx] + wrap_angle(bounded - q[:, idx])
        else:
            bounds = j.get("knee_hard_limits", {}).get(name)
            if bounds is None:
                targets[:, column] = desired[:, column]
            else:
                lo, hi = bounds
                margin = 0.0 if is_round2(contract) else j.get("knee_soft_margin", 0.0)
                targets[:, column] = desired[:, column].clamp(lo + margin, hi - margin)
    wheel = clipped[:, j["wheel_indices"]] * a["wheel_velocity_scale"]
    return targets, wheel, clipped


def motor_torque_limit(joint_speed, wheel_config):
    """按 gear_ratio*joint_speed 查电机侧曲线，再折算回轮端力矩。"""
    if not isinstance(joint_speed, torch.Tensor) or not joint_speed.is_floating_point() or not torch.isfinite(joint_speed).all():
        raise ValueError("joint_speed must be a finite floating tensor")
    ratio = wheel_config["gear_ratio"]
    eta = wheel_config["gearbox_efficiency"]
    if ratio <= 0 or not 0 < eta <= 1 or wheel_config["curve_side"] != "motor":
        raise ValueError("Invalid motor-side curve domain")
    speeds = _like(wheel_config["motor_speed_rad_s"], joint_speed)
    torques = _like(wheel_config["motor_torque_nm"], joint_speed)
    if (speeds.ndim != 1 or speeds.numel() < 2 or speeds.numel() != torques.numel()
            or not (speeds[1:] > speeds[:-1]).all() or not torch.isfinite(speeds).all()
            or not torch.isfinite(torques).all() or (torques < 0).any()):
        raise ValueError("Invalid torque-speed samples")
    query = joint_speed.abs().reshape(-1).contiguous() * ratio
    idx = torch.searchsorted(speeds, query).clamp(1, speeds.numel() - 1)
    low, high = idx - 1, idx
    weight = ((query - speeds[low]) / (speeds[high] - speeds[low])).clamp(0, 1)
    motor_torque = torques[low] + weight * (torques[high] - torques[low])
    motor_torque = torch.where(query > speeds[-1], torch.zeros_like(motor_torque), motor_torque)
    return (motor_torque * ratio * eta).clamp(max=wheel_config["effort_limit"]).reshape(joint_speed.shape)


def compute_torques(joint_pos6, joint_vel6, leg_targets4, wheel_targets2, contract):
    q = _matrix(joint_pos6, 6, "joint positions")
    n = q.shape[0]
    v = _matrix(joint_vel6, 6, "joint velocities", n)
    legs = _matrix(leg_targets4, 4, "leg targets", n)
    wheels = _matrix(wheel_targets2, 2, "wheel targets", n)
    ids, actuators = contract["joints"], contract["actuators"]
    leg_cfg, wheel_cfg = actuators["leg"], actuators["wheel"]
    torque = torch.zeros_like(q)
    leg_torque = leg_cfg["kp"] * (legs - q[:, ids["leg_indices"]]) - leg_cfg["kd"] * v[:, ids["leg_indices"]]
    torque[:, ids["leg_indices"]] = leg_torque.clamp(-leg_cfg["effort_limit"], leg_cfg["effort_limit"])
    wheel_torque = wheel_cfg["kd"] * (wheels - v[:, ids["wheel_indices"]])
    bound = motor_torque_limit(v[:, ids["wheel_indices"]], wheel_cfg)
    torque[:, ids["wheel_indices"]] = torch.maximum(torch.minimum(wheel_torque, bound), -bound)
    if not torch.isfinite(torque).all():
        raise ValueError("Nonfinite actuator effort")
    return torque


def compute_reward_terms(v_body3, w_body3, gravity3, height, commands3, actions6,
                         previous_actions6, torques6, joint_pos6, contract,
                         wheel_clearance2=None, wheel_slip2=None, wheel_contact2=None,
                         leg_joint_vel4=None, leg_joint_vel_ema4=None,
                         previous_previous_actions6=None):
    """加权、按 policy_dt 缩放的即时奖励（不要重复乘 dt）。

    新增的防弹跳输入全部可选：``wheel_clearance2`` 为轮心相对地面高度减去
    (轮半径+容差)（>0 表示离地），``wheel_slip2`` 为轮底切向滑移速度平方，
    ``wheel_contact2`` 为触地浮点标志，``leg_joint_vel4``/``leg_joint_vel_ema4``
    用于腿关节振荡惩罚，``previous_previous_actions6`` 用于腿动作二阶平滑。
    """
    v = _matrix(v_body3, 3, "body linear velocity")
    n = v.shape[0]
    for value, width, name in [(w_body3, 3, "body angular velocity"), (gravity3, 3, "gravity"),
                                (commands3, 3, "commands"), (actions6, 6, "actions"),
                                (previous_actions6, 6, "previous actions"), (torques6, 6, "torques"),
                                (joint_pos6, 6, "joint positions")]:
        _matrix(value, width, name, n)
    for value, width, name in [(wheel_clearance2, 2, "wheel clearance"),
                               (wheel_slip2, 2, "wheel slip"), (wheel_contact2, 2, "wheel contact"),
                               (leg_joint_vel4, 4, "leg joint velocity"),
                               (leg_joint_vel_ema4, 4, "leg joint velocity EMA"),
                               (previous_previous_actions6, 6, "previous previous actions")]:
        if value is not None:
            _matrix(value, width, name, n)
    h = _vector(height, n, "base height")
    r, j = contract["rewards"], contract["joints"]
    soft = torch.zeros_like(h)
    hard = j.get("knee_hard_limits", {})
    for idx in j["knee_indices"]:
        name = j["action_order"][idx]
        if name not in hard:
            continue
        lo, hi = hard[name]
        if is_round2(contract):
            margin = (hi - lo) * (1 - j["soft_position_limit_factor"]) / 2
            soft = soft + (lo + margin - joint_pos6[:, idx]).clamp(min=0)
            soft = soft + (joint_pos6[:, idx] - hi + margin).clamp(min=0)
        else:
            margin = j.get("knee_soft_margin", 0.0)
            soft = soft + (lo + margin - joint_pos6[:, idx]).clamp(min=0).square()
            soft = soft + (joint_pos6[:, idx] - hi + margin).clamp(min=0).square()
    caps = [contract["actuators"]["wheel" if idx in j["wheel_indices"] else "leg"]["effort_limit"] for idx in range(6)]
    # 接触门控：轮子离地时不再给速度/偏航追踪奖励（掐断"腾空拿速度分"）。
    if wheel_clearance2 is not None and bool(r.get("velocity_contact_gate", False)):
        gate_sigma = max(float(r.get("contact_gate_sigma_m", 0.01)), 1e-6)
        contact_gate = torch.exp(-wheel_clearance2.clamp(min=0.0) / gate_sigma).prod(dim=-1)
    else:
        contact_gate = torch.ones_like(h)
    raw = {
        "velocity": contact_gate * torch.exp(-((v[:, 0] - commands3[:, 0]) / r["sigma_velocity"]).square()),
        "yaw": contact_gate * torch.exp(-((w_body3[:, 2] - commands3[:, 1]) / r["sigma_yaw"]).square()),
        "height": torch.exp(-((h - commands3[:, 2]) / r["sigma_height"]).square()),
        "upright": gravity3[:, :2].square().sum(-1),
        "lateral_velocity": v[:, 1].square(), "vertical_velocity": v[:, 2].square(),
        "action_rate": (actions6 - previous_actions6).square().sum(-1),
        "effort": (torques6 / _like(caps, torques6)).square().sum(-1),
        "knee_soft_limit": soft,
        "zero_command_translation": (commands3[:, 0].abs() < r["zero_vx_threshold"]).to(v.dtype) * v[:, :2].square().sum(-1),
    }
    if wheel_clearance2 is not None:
        raw["wheel_hop"] = wheel_clearance2.clamp(min=0.0).square().sum(-1)
    if wheel_slip2 is not None and wheel_contact2 is not None:
        raw["wheel_slip"] = (wheel_slip2 * (wheel_contact2 > 0.0).to(wheel_slip2.dtype)).sum(-1)
    if leg_joint_vel4 is not None and leg_joint_vel_ema4 is not None:
        raw["leg_joint_osc"] = (leg_joint_vel4 - leg_joint_vel_ema4).square().sum(-1)
    if previous_previous_actions6 is not None:
        second_diff = actions6 - 2.0 * previous_actions6 + previous_previous_actions6
        raw["action_smoothness_leg"] = second_diff[:, j["leg_indices"]].square().sum(-1)
    return {name: value * r["weights"][name] * contract["timing"]["policy_dt"] for name, value in raw.items()}
