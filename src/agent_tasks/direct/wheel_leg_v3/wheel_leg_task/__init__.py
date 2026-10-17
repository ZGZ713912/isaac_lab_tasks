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

from agent_tasks.direct.wheel_leg_v3 import agents  # noqa: F401

gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-v1",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatEnvCfg_v1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-v2",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatEnvCfg_v2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-Play-v2",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatEnvCfg_v2_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Rough-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3RoughPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Rough-v1",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3RoughEnvCfg_v1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3RoughPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-DreamWaQ-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatDreamWaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatDreamWaqPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-HIM-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatHIMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatHIMPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-NP3OBarlow-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatNP3OBarlowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatNP3OBarlowPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-DreamWaQ-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatDreamWaqEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatDreamWaqPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-HIM-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatHIMEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatHIMPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Flat-NP3OBarlow-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3FlatNP3OBarlowEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3FlatNP3OBarlowPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Rough-Play-v0",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3RoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3RoughPPORunnerCfg",
    },
)


gym.register(
    id="Robotics-Wheel-Leg-V3-Rough-Play-v1",
    entry_point=f"{__name__}.env:WheelLegV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelLegV3RoughEnvCfg_v1_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLegV3RoughPPORunnerCfg",
    },
)

