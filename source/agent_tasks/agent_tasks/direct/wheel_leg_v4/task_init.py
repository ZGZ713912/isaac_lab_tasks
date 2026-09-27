import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import WheelLegV4PPORunnerCfg


if "Robotics-Wheel-Leg-V4-Flat-v0" not in gym.registry:
    gym.register(
        id="Robotics-Wheel-Leg-V4-Flat-v0",
        entry_point="agent_tasks.direct.wheel_leg_v4.env:WheelLegV4Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "agent_tasks.direct.wheel_leg_v4.env_cfg:WheelLegV4FlatEnvCfg",
            "rsl_rl_cfg_entry_point": "agent_tasks.direct.wheel_leg_v4.agents.rsl_rl_ppo_cfg:WheelLegV4PPORunnerCfg",
        },
    )

if "Robotics-Wheel-Leg-V4-Flat-Play-v0" not in gym.registry:
    gym.register(
        id="Robotics-Wheel-Leg-V4-Flat-Play-v0",
        entry_point="agent_tasks.direct.wheel_leg_v4.env:WheelLegV4Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "agent_tasks.direct.wheel_leg_v4.env_cfg:WheelLegV4FlatPlayEnvCfg",
            "rsl_rl_cfg_entry_point": "agent_tasks.direct.wheel_leg_v4.agents.rsl_rl_ppo_cfg:WheelLegV4PPORunnerCfg",
        },
    )
