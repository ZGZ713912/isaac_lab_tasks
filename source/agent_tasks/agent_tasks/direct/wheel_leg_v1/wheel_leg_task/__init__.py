# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

import gymnasium as gym

from agent_tasks.direct.wheel_leg_v1 import agents  # noqa: F401

gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-v1",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatEnvCfg_v1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-v2",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatEnvCfg_v2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-Play-v2",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatEnvCfg_v2_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Rough-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Rough-v1",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1RoughEnvCfg_v1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-DreamWaQ-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatDreamWaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatDreamWaqPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-HIM-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatHIMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatHIMPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-NP3OBarlow-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatNP3OBarlowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatNP3OBarlowPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-DreamWaQ-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatDreamWaqEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatDreamWaqPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-HIM-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatHIMEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatHIMPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Flat-NP3OBarlow-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1FlatNP3OBarlowEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1FlatNP3OBarlowPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Rough-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1RoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1RoughPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V1-Rough-Play-v1",
    entry_point=f"{__name__}.env:WheelLegV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV1RoughEnvCfg_v1_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV1RoughPPORunnerCfg",
    },
)

