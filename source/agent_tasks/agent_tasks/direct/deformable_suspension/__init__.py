# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# DeformableSuspension 任务注册。
# 合同（与 RMCS rmcs_rl 部署逐项同构）：
#   obs 22 = cmd3 | height_cmd1 | ang_vel3 | gravity3 | leg_pos4 | leg_vel4 | act4
#   act  4 = 腿关节位置 PD 目标
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
