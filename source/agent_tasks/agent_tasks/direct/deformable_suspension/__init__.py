# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# DeformableSuspension 任务注册。
# 合同（单层重写版）：
#   obs 26 = q_cmd1 | cmd3(vx,vy,ωz) | ang_vel_b3 | proj_grav_b3
#            | leg_pos4(绝对角) | leg_vel4 | leg_torque4 | act4
#   critic 34 = obs26 + lin_vel_b3 + 真实车高1 + 四轮接触力4
#   act   4 = joint_leg_* 位置目标（腿级联 PID：外环位置 PI→速度指令，内环速度 PI→力矩）
# 轮子为球体碰撞（无牵引），运动由外部底盘速度伺服实现（首版静态关闭）。
# =============================================================================

import gymnasium as gym

from agent_tasks.direct.deformable_suspension import agents

gym.register(
    id="Robotics-Deformable-Suspension-v0",
    entry_point=f"{__name__}.env:DeformableSuspensionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DeformableSuspensionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableSuspensionPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Deformable-Suspension-Rough-v0",
    entry_point=f"{__name__}.env:DeformableSuspensionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DeformableSuspensionRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableSuspensionPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Deformable-Suspension-Rough-Steep-v0",
    entry_point=f"{__name__}.env:DeformableSuspensionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DeformableSuspensionRoughSteepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableSuspensionPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Deformable-Suspension-Play-v0",
    entry_point=f"{__name__}.env:DeformableSuspensionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DeformableSuspensionFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableSuspensionPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Deformable-Suspension-Rough-Play-v0",
    entry_point=f"{__name__}.env:DeformableSuspensionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DeformableSuspensionRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableSuspensionPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Deformable-Suspension-Rough-Steep-Play-v0",
    entry_point=f"{__name__}.env:DeformableSuspensionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DeformableSuspensionRoughSteepPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableSuspensionPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Deformable-Suspension-Rough-Keyboard-Play-v0",
    entry_point=f"{__name__}.env:DeformableSuspensionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DeformableSuspensionRoughKeyboardPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableSuspensionPPORunnerCfg",
    },
)

gym.register(
    id="Robotics-Deformable-Suspension-Rough-Steep-Keyboard-Play-v0",
    entry_point=f"{__name__}.env:DeformableSuspensionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DeformableSuspensionRoughSteepKeyboardPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableSuspensionPPORunnerCfg",
    },
)
