# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
#
# Wheel_leg_V2 周期坡面地形的 Torch 求高工具。
#
# 与 agent_world.terrains.periodic_slope_height 同源（同种子/同公式），
# 保证地形网格与训练时的解析地面高度完全一致，无量化误差。
# 参考：deformable_suspension/cfg_utils.py 的同名实现。
# =============================================================================

from __future__ import annotations

import math

import torch

from agent_world.terrains import periodic_slope_angle


def build_periodic_slope_angle_table(
    segment_length: float,
    angle_range: tuple[float, float],
    seed: int,
    num_periods: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """预生成坡角表（每周期一个角，度），供 torch 求高用。"""
    return torch.tensor(
        [periodic_slope_angle(k, angle_range, seed) for k in range(int(num_periods))],
        dtype=torch.float32,
        device=device,
    )


def periodic_slope_height_torch(
    x: torch.Tensor, segment_length: float, angle_table: torch.Tensor
) -> torch.Tensor:
    """周期坡面在 world x 处的地面高度（m），与 numpy 版逐点一致。"""
    seg = float(segment_length)
    period = 4.0 * seg
    k = torch.floor(x / period).long().clamp_(0, angle_table.numel() - 1)
    slope = torch.tan(angle_table[k] * (math.pi / 180.0))
    s = x - k.to(x.dtype) * period
    h = torch.zeros_like(x)
    h = torch.where((s >= 0.0) & (s < seg), slope * s, h)
    h = torch.where((s >= seg) & (s < 2.0 * seg), slope * seg, h)
    h = torch.where((s >= 2.0 * seg) & (s < 3.0 * seg), slope * (3.0 * seg - s), h)
    return h


def periodic_slope_gradient_torch(
    x: torch.Tensor, segment_length: float, angle_table: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    """周期坡面的 dh/dx（解析差分），用于求地面法向与 spawn 姿态对齐。"""
    forward = periodic_slope_height_torch(x + eps, segment_length, angle_table)
    backward = periodic_slope_height_torch(x - eps, segment_length, angle_table)
    return (forward - backward) / (2.0 * eps)
