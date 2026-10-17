"""Necessary static wheel-force margin for training reward diagnostics.

This ideal driven-wheel bound ignores yaw balance, transient wheel inertia and
lateral forces. A positive margin is not a guarantee of whole-vehicle tracking.
It is derived from simulator state for rewards, never an actor observation.
"""
from __future__ import annotations

import torch


def uphill_drive_margin(
    normal_force: torch.Tensor,
    roll_direction_w: torch.Tensor,
    ground_normal_w: torch.Tensor,
    friction: torch.Tensor,
    *,
    mass_kg: float,
    wheel_torque_limit_nm: float,
    wheel_radius_m: float,
    reserve_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return squared deficit, ideal uphill drive capacity and uphill gravity.

    Forces are (N, 4), directions (N, 4, 3), ground normals (N, 3) and
    friction (N, 1). The plane normal comes from the existing terrain model.
    """
    if mass_kg <= 0 or wheel_radius_m <= 0 or wheel_torque_limit_nm < 0 or reserve_fraction < 0:
        raise ValueError("Positive mass/radius and nonnegative torque/reserve required")
    normal = torch.nn.functional.normalize(ground_normal_w, dim=-1)
    vertical = torch.zeros_like(normal)
    vertical[:, 2] = 1.0
    tangent = vertical - normal[:, 2:3] * normal
    sine_grade = tangent.norm(dim=-1)
    uphill = tangent / sine_grade[:, None].clamp_min(1.e-7)
    projection = (roll_direction_w * uphill[:, None]).sum(-1).abs()
    motor_force = normal_force.new_tensor(wheel_torque_limit_nm / wheel_radius_m)
    traction = torch.minimum(friction.clamp_min(0.0) * normal_force.clamp_min(0.0), motor_force)
    capacity = (traction * projection).sum(-1)
    gravity = mass_kg * 9.81 * sine_grade
    deficit = ((1.0 + reserve_fraction) * gravity - capacity).clamp_min(0.0)
    cost = (deficit / gravity.clamp_min(10.0)).square()
    return cost, capacity, gravity
