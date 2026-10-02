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
# 当前训练阶段固定低车身，运动伺服关闭；先训练静态主动调平。
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

# ---- Transformer 策略变体（env 相同，仅网络结构不同）----
_TRANSFORMER_RUNNER = f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableSuspensionTransformerPPORunnerCfg"
for _task_id, _env_cfg in (
    ("Robotics-Deformable-Suspension-Rough-Transformer-v0", "DeformableSuspensionRoughEnvCfg"),
    ("Robotics-Deformable-Suspension-Rough-Transformer-Play-v0", "DeformableSuspensionRoughPlayEnvCfg"),
    (
        "Robotics-Deformable-Suspension-Rough-Transformer-Keyboard-Play-v0",
        "DeformableSuspensionRoughKeyboardPlayEnvCfg",
    ),
):
    gym.register(
        id=_task_id,
        entry_point=f"{__name__}.env:DeformableSuspensionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{_env_cfg}",
            "rsl_rl_cfg_entry_point": _TRANSFORMER_RUNNER,
        },
    )

# V1 changes the observation/action contract and must not load V0 checkpoints.
for _suffix, _cfg in (
    ("Flat", "DeformableDynamicFlatEnvCfg"),
    ("Rough", "DeformableDynamicRoughEnvCfg"),
    ("Rough-Steep", "DeformableDynamicRoughSteepEnvCfg"),
    ("Rough-Keyboard-Play", "DeformableDynamicKeyboardPlayEnvCfg"),
):
    gym.register(
        id=f"Robotics-Deformable-Suspension-{_suffix}-History-Transformer-v1",
        entry_point=f"{__name__}.dynamic_env:DeformableDynamicEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.dynamic_cfg:" + {
                "Flat": "DeformableLegacyFlatEnvCfg", "Rough": "DeformableLegacyRoughEnvCfg",
                "Rough-Steep": "DeformableLegacySteepEnvCfg", "Rough-Keyboard-Play": "DeformableLegacyKeyboardEnvCfg",
            }[_suffix],
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableLegacyPPORunnerCfg",
        },
    )

    gym.register(
        id=f"Robotics-Deformable-Suspension-{_suffix}-History-Transformer-v2",
        entry_point=f"{__name__}.dynamic_env:DeformableDynamicEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.dynamic_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeformableDynamicPPORunnerCfg",
        },
    )
