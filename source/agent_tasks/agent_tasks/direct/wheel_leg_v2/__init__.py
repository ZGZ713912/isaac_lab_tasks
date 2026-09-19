# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Wheel_leg_V2（闭链轮腿）任务注册。
#
# 合同（与 V40 一致，仅驱动关节/资产不同）：
#   policy 125 = 5 × 25
#     25 = ang_vel_b3 | proj_grav_b3 | cmd(vx,wz,height)3
#          | 四腿相对名义角4 | 六关节速度6 | 上一动作6
#   critic 29 = 25 + 真值线速度3 + 真实车高1
#   action  6 = 4 个电机腿关节位置残差 + 2 个轮速度目标
# =============================================================================

import gymnasium as gym

from agent_tasks.direct.wheel_leg_v2 import agents  # noqa: F401


def _register(task_id: str, cfg_name: str) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.env:WheelLegV2Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV2PPORunnerCfg",
        },
    )


_register("Robotics-Wheel-Leg-V2-Stand-v0", "WheelLegV2StandEnvCfg")
_register("Robotics-Wheel-Leg-V2-Height-v0", "WheelLegV2HeightEnvCfg")
_register("Robotics-Wheel-Leg-V2-Flat-v0", "WheelLegV2FlatEnvCfg")
_register("Robotics-Wheel-Leg-V2-Flat-Round2-v0", "WheelLegV2FlatRound2EnvCfg")
_register("Robotics-Wheel-Leg-V2-Stand-Play-v0", "WheelLegV2StandPlayEnvCfg")
_register("Robotics-Wheel-Leg-V2-Flat-Play-v0", "WheelLegV2FlatPlayEnvCfg")
_register("Robotics-Wheel-Leg-V2-Flat-Round2-Play-v0", "WheelLegV2FlatRound2PlayEnvCfg")
