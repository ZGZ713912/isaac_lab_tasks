#!/usr/bin/env bash
# =============================================================================
# 02 - 仓库必需补丁（全新 clone 一定缺这两个，不修跑不起来）
# 用法：cd ~/wheeled-legged_RL && conda activate isaaclab && bash 02_fixes.sh
#
# 1) 重建缺失的 source/agent_rl/agent_rl/rsl_rl/env/ 包
#    （被 .gitignore 的通用 "env/" 规则误伤，从未入库；所有任务 import 都需要它）
# 2) 修入口脚本的 sys.path，让 python scripts/rsl_rl/train.py 能 import scripts.utils
# =============================================================================
set -euo pipefail
REPO_ROOT="$(pwd)"
ENVDIR="$REPO_ROOT/source/agent_rl/agent_rl/rsl_rl/env"

echo "[1/2] 重建缺失的 env 包 -> $ENVDIR"
mkdir -p "$ENVDIR"

cat > "$ENVDIR/__init__.py" <<'PYEOF'
from .vec_env import VecEnv
from .rsl_rl_vec_env import RslRlVecEnvWrapper

__all__ = ["VecEnv", "RslRlVecEnvWrapper"]
PYEOF

cat > "$ENVDIR/vec_env.py" <<'PYEOF'
"""Abstract vectorized environment interface for RSL-RL style runners."""
from __future__ import annotations
import torch
from abc import ABC, abstractmethod


class VecEnv(ABC):
    num_envs: int
    num_actions: int
    max_episode_length: int | torch.Tensor
    episode_length_buf: torch.Tensor
    device: torch.device
    cfg: dict | object
    num_observations: dict[str, int]
    num_privileged_obs: int | None

    @abstractmethod
    def get_observations(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
PYEOF

cat > "$ENVDIR/rsl_rl_vec_env.py" <<'PYEOF'
"""RSL-RL wrapper around an Isaac Lab env, for the custom runners (DreamWaQ/HIM/NP3O)."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict
from .vec_env import VecEnv


class RslRlVecEnvWrapper(VecEnv):
    def __init__(self, env, clip_actions: float | None = None):
        from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv  # lazy: needs SimulationApp
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(f"Unsupported env type: {type(env)}")
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        self._modify_action_space()
        obs_dict, _ = self.env.reset()
        self.num_observations = {k: int(np.prod(v.shape[1:])) for k, v in obs_dict.items()}
        self.num_privileged_obs = self._resolve_privileged_obs_dim()

    def _resolve_privileged_obs_dim(self):
        state_space = getattr(self.unwrapped.cfg, "state_space", None)
        if isinstance(state_space, dict):
            d = state_space.get("critic", None)
            return int(d) if isinstance(d, int) and d > 0 else None
        if isinstance(state_space, int) and state_space > 0:
            return state_space
        return None

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self):
        obs_dict, extras = self.env.reset()
        return self._as_obs_dict(obs_dict), extras

    def get_observations(self):
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._as_obs_dict(obs_dict)

    def step(self, actions):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self._as_obs_dict(obs_dict), rew, dones, extras

    def close(self):
        return self.env.close()

    @staticmethod
    def _as_obs_dict(obs_dict):
        if isinstance(obs_dict, TensorDict):
            return {k: v for k, v in obs_dict.items()}
        return dict(obs_dict)

    def _modify_action_space(self):
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
PYEOF

echo "     已创建: $(ls "$ENVDIR")"

echo "[2/2] 给入口脚本加 sys.path 补丁（幂等）"
PYTHON="$HOME/miniconda3/envs/isaaclab/bin/python"
for f in scripts/rsl_rl/train.py scripts/rsl_rl/play.py scripts/view_robot.py scripts/list_envs.py; do
    if ! grep -q "_REPO_ROOT" "$REPO_ROOT/$f" 2>/dev/null; then
        echo "     打补丁: $f"
        "$PYTHON" - "$REPO_ROOT/$f" <<'PY'
import sys
path = sys.argv[1]
src = open(path).read()
anchor = '"""Launch Isaac Sim Simulator first."""'
assert anchor in src, f"anchor not found in {path}"
# train.py / play.py 在 scripts/rsl_rl/ 下 -> dirname*3；view_robot/list_envs 在 scripts/ 下 -> dirname*2
depth = 3 if path.rsplit("/", 1)[0].endswith("rsl_rl") else 2
parents = "/".join([".."] * depth)
inject = anchor + f'''

import os
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), {parents!r}))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)'''
open(path, "w").write(src.replace(anchor, inject, 1))
print("  patched", path)
PY
    else
        echo "     已打过: $f (跳过)"
    fi
done

echo ""
echo "============================================================"
echo "  补丁完成。下一步: bash 03_smoke.sh  (无头冒烟验证)"
echo "  运行训练（记住加 PYTHONPATH）:"
echo "  cd ~/wheeled-legged_RL && conda activate isaaclab"
echo "  PYTHONPATH=$PWD python scripts/rsl_rl/train.py --task=Robotics-Wheelbipe-V14-Flat-v0 \\"
echo "      --num_envs=4096 --max_iterations=20000 --headless"
echo "============================================================"
