# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V3（Wheel_leg_V2 闭链机器人）任务家族包（4 层复刻自 wheelbipe）：
#   wheel_leg_base/    ← wheelbipe25_v3（RL 基座：观测/奖励/终止/随机化/噪声/课程）
#   wheel_leg_terrain/ ← wheelbipe_V13（地形命令管理器）
#   wheel_leg_task/    ← wheelbipe_V14（任务注册层，去云台；闭链四杆 + 气簧在 base 侧适配）
#
# 机器人：Wheel_leg_V2（4 spherical 闭链 + 2 prismatic 气簧，无云台），执行器用 IdealPD。
# 控制频率 100Hz（sim.dt=1/200, decimation=2）。
# =============================================================================
