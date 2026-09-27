"""Explicit V5-style joint-space control for Wheel_leg_V4."""

import torch


def _wheel_limit(speed: torch.Tensor) -> torch.Tensor:
    # V5 reference prior: motor-side linearly decreasing envelope, mapped through
    # the 11:1 gearbox. The source is intentionally explicit and replaceable.
    ratio = 11.0
    motor_speed = speed.abs() * ratio
    max_motor_speed = 999.0630117647059
    motor_stall_torque = 0.34888059701492535
    available = motor_stall_torque * (1.0 - motor_speed / max_motor_speed)
    return (available.clamp(min=0.0) * ratio).clamp(max=3.837686567164179)


def compute_v4_torques(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    leg_targets: torch.Tensor,
    wheel_targets: torch.Tensor,
    *,
    leg_kp: float,
    leg_kd: float,
    wheel_kd: float,
    wheel_effort_limit: float,
) -> torch.Tensor:
    if joint_pos.shape[-1] != 6 or joint_vel.shape != joint_pos.shape:
        raise ValueError("V4 joint state must have shape [N, 6]")
    torque = torch.zeros_like(joint_pos)
    leg_ids = (0, 1, 2, 3)
    wheel_ids = (4, 5)
    leg = leg_kp * (leg_targets - joint_pos[:, leg_ids]) - leg_kd * joint_vel[:, leg_ids]
    torque[:, leg_ids] = leg.clamp(-40.0, 40.0)
    wheel = wheel_kd * (wheel_targets - joint_vel[:, wheel_ids])
    bound = _wheel_limit(joint_vel[:, wheel_ids]).clamp(max=wheel_effort_limit)
    torque[:, wheel_ids] = torch.minimum(torch.maximum(wheel, -bound), bound)
    return torch.nan_to_num(torque, nan=0.0, posinf=0.0, neginf=0.0)
