# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# DeformableSuspension 任务的标定常量与工具（照 wheelbipe_V14/cfg_utils.py 的
# “常量表 + 读取/应用辅助函数”风格，但几何部分换成平四的腿角↔高度标定，
# 不涉及五连杆 IK / 云台 / 弹簧）。
#
# 关键几何事实（2026-09-16 由 URDF 正运动学实测，见 docs 与本文件常量）：
#   - 平四腿角 q ∈ [0, 1.36] rad；q 越大车体越低、轮距越大（与旧 V1 相反）。
#   - q=0        → base 原点离地 0.13189 m（高基准，初始 0°；底盘网格底离地 0.1049）
#   - q=1.0563   → base 原点离地 0.0370 m（低基准：底盘网格最低点离地 ≈1cm）
#   - 车体网格在 base 系 Z ∈ [-0.027, +0.206]：底盘底 = 离地 − 0.027，车顶 = 离地 + 0.206
#     → 高基准车顶 ≈0.338 m（超 260mm 隧道），低基准 ≈0.243 m（可过隧道）
#   - 注意：q 再大（≈1.30 以上）base 原点会低于轮底接触面，底盘必然插地；
#     因此“最低可用的低基准”由底盘网格离地决定（1cm ↔ q=1.0563）。
#   - 四全向轮：轮心 (±0.2141, ±0.2141, -0.055)，轮轴沿对角线(±45°/±135°)，
#     标准 X 型布局；轮半径 0.0769。碰撞体为解析球体，无法传递牵引力，
#     故“动态运动”由外部底盘速度伺服实现，轮速由球心速度投影计算。
# =============================================================================

from __future__ import annotations

import math

import numpy as np
import torch

from agent_world.terrains import periodic_slope_angle

# ---------------------------------------------------------------------------
# 1. 腿角 ↔ 车高标定
# ---------------------------------------------------------------------------
# q ∈ [0,1.36] → base 原点离地高度（m），5 次多项式拟合（max err 0.0009 mm）。
# 由 URDF 正运动学：h(q) = R_wheel - z_wheel_center(q)。
_Q_TO_HEIGHT_COEFFS = (
    -0.000890116,
    0.006623633,
    0.004219653,
    -0.068262985,
    -0.029135306,
    0.131890850,
)

LEG_LOWER_LIMIT = 0.0
LEG_UPPER_LIMIT = 1.36

# 两档基准腿角与对应离地高度（首版只用这两档；q_cmd 以连续量下发）
# 低基准按“底盘网格最低点离地 1cm”标定（q=1.254 会让底盘插地 1.7cm，已弃用）。
Q_HIGH = 0.0
Q_LOW = 1.0563
H_HIGH = 0.13189
H_LOW = 0.03700

# 车体几何：底盘网格最低/最高点相对 base 原点的偏移（m）
BODY_BOTTOM_OFFSET = -0.027
BODY_TOP_OFFSET = 0.206
TUNNEL_MAX_BODY_TOP = 0.260  # 隧道净高任务的车顶上限（低基准 0.243，留 1.7cm 余量）


def q_to_base_height(q: torch.Tensor | float) -> torch.Tensor | float:
    """平四腿角 → base 原点离地高度（m）。支持 tensor/标量。"""
    if isinstance(q, torch.Tensor):
        out = torch.zeros_like(q)
        for c in _Q_TO_HEIGHT_COEFFS:
            out = out * q + c
        return out
    return float(np.polyval(_Q_TO_HEIGHT_COEFFS, q))


def q_to_body_top(q: torch.Tensor | float) -> torch.Tensor | float:
    """平四腿角 → 车顶相对地面高度（m），用于隧道约束判定。"""
    return q_to_base_height(q) + BODY_TOP_OFFSET


# ---------------------------------------------------------------------------
# 2. 四全向轮几何（X 型 45° 布局）
# ---------------------------------------------------------------------------
WHEEL_RADIUS = 0.0769
WHEEL_HALF_BASE = 0.2141  # 轮心 |x| = |y|
_S = float(np.sqrt(0.5))  # sin/cos 45° = 0.7071...

# 轮心在 base 系的位置（与关节顺序 joint_wheel_1..4 对应）
WHEEL_CENTER_XY = (
    (+WHEEL_HALF_BASE, -WHEEL_HALF_BASE),
    (+WHEEL_HALF_BASE, +WHEEL_HALF_BASE),
    (-WHEEL_HALF_BASE, +WHEEL_HALF_BASE),
    (-WHEEL_HALF_BASE, -WHEEL_HALF_BASE),
)
# 各轮“地面内滚动方向”（单位向量，垂直于轮轴）
WHEEL_ROLL_DIRS = (
    (+_S, +_S),
    (-_S, +_S),
    (-_S, -_S),
    (+_S, -_S),
)
# 全向逆运动学：ω_i = (1/r)(u_i·v_xy + OMNI_YAW_COEFF·ω_z)
# OMNI_YAW_COEFF = (r_i × u_i)_z = 2·L·s，四轮同值
OMNI_YAW_COEFF = 2.0 * WHEEL_HALF_BASE * _S  # ≈0.30276

# 关节/连杆顺序（与 assets/deformable_V2.py 的 URDF 顺序一致）
ORDERED_LEG_JOINT_NAMES = ("joint_leg_1", "joint_leg_2", "joint_leg_3", "joint_leg_4")
ORDERED_WS_JOINT_NAMES = (
    "joint_wheel_set_1",
    "joint_wheel_set_2",
    "joint_wheel_set_3",
    "joint_wheel_set_4",
)
ORDERED_UPPER_LEG_JOINT_NAMES = (
    "joint_upper_leg_1",
    "joint_upper_leg_2",
    "joint_upper_leg_3",
    "joint_upper_leg_4",
)
ORDERED_WHEEL_JOINT_NAMES = ("joint_wheel_1", "joint_wheel_2", "joint_wheel_3", "joint_wheel_4")


def sphere_roll_speeds(
    base_lin_vel_b: torch.Tensor, base_ang_vel_b: torch.Tensor, radius: float = WHEEL_RADIUS
) -> torch.Tensor:
    """由车体速度推算四轮“等效滚动线速度”（m/s），形状 (N,4)。

    球体碰撞无法用关节编码器反映真实滚动，故用轮心处车身速度在地面内投影：
        v_i = v_body + ω_body × r_i
        v_roll_i = v_i · u_i   （u_i = 地面内滚动方向）
    返回的等效轮角速度 = v_roll_i / radius。

    注意：车体系下 ω × r 的 z 分量不影响地面内投影（r_i 的 z 为 -0.055，
    ω_xy × r 会引入面内分量；这里保留完整 3D 叉乘后取 xy，再投到 u_i）。
    """
    n = base_lin_vel_b.shape[0]
    device = base_lin_vel_b.device
    r = torch.zeros(n, 4, 3, dtype=base_lin_vel_b.dtype, device=device)
    u = torch.zeros(n, 4, 2, dtype=base_lin_vel_b.dtype, device=device)
    for i in range(4):
        r[:, i, 0] = WHEEL_CENTER_XY[i][0]
        r[:, i, 1] = WHEEL_CENTER_XY[i][1]
        r[:, i, 2] = -0.055
        u[:, i, 0] = WHEEL_ROLL_DIRS[i][0]
        u[:, i, 1] = WHEEL_ROLL_DIRS[i][1]
    v = base_lin_vel_b.unsqueeze(1) + torch.cross(
        base_ang_vel_b.unsqueeze(1).expand(-1, 4, -1), r, dim=-1
    )
    return (v[..., :2] * u).sum(dim=-1)


# ---------------------------------------------------------------------------
# 3. 观测缩放 / 裁剪表（照 wheelbipe 的 block 表写法）
# ---------------------------------------------------------------------------
# policy 观测块顺序：q_cmd | cmd | ang_vel | gravity | leg_pos | leg_vel | leg_torque | act
OBS_SCALE = {
    "q_cmd": 1.0,  # rad
    "cmd": 1.0,  # m/s, m/s, rad/s
    "ang_vel": 0.5,  # rad/s
    "gravity": 1.0,
    "leg_pos": 1.0,  # rad（绝对角，不用相对量）
    "leg_vel": 0.1,  # rad/s
    "leg_torque": 0.05,  # N·m（40 N·m → 2.0）
    "act": 1.0,
}

OBS_CLIP = {
    "q_cmd": (-1.0, 2.0),
    "cmd": (-3.0, 3.0),
    "ang_vel": (-10.0, 10.0),
    "gravity": (-1.0, 1.0),
    "leg_pos": (-0.5, 2.0),
    "leg_vel": (-30.0, 30.0),
    "leg_torque": (-60.0, 60.0),
    "act": (-1.5, 1.5),
}


# ---------------------------------------------------------------------------
# 4. 周期坡面地形（Torch 版；与 agent_world.terrains.periodic_slope_height 同源）
# ---------------------------------------------------------------------------
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


def periodic_slope_flat_mask(x: torch.Tensor, segment_length: float) -> torch.Tensor:
    """当前位置是否处于平路段（[L,2L) 或 [3L,4L)）。"""
    seg = float(segment_length)
    period = 4.0 * seg
    s = x - torch.floor(x / period) * period
    return ((s >= seg) & (s < 2.0 * seg)) | (s >= 3.0 * seg)
