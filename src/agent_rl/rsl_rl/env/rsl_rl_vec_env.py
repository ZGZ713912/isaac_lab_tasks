# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [Isaac Lab]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2022-2026, The Isaac Lab Project Developers
# =============================================================================

"""RSL-RL vectorized environment wrapper around an Isaac Lab environment.

This is the wrapper used by the custom runners (DreamWaQ / HIM / NP3O) via
``train.py``::

    from agent_rl.rsl_rl.env import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

It is equivalent to ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`` plus the
``num_observations`` / ``num_privileged_obs`` bookkeeping that the custom
runners require (plain ``dict[str, int]`` observation dims).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from .vec_env import VecEnv


class RslRlVecEnvWrapper(VecEnv):
    """Wraps an Isaac Lab environment for RSL-RL style runners.

    .. caution::
        This class must be the last wrapper in the wrapper chain. It does not
        follow the :class:`gym.Wrapper` interface.
    """

    def __init__(self, env, clip_actions: float | None = None):
        """Initialize the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL
            runners do not call reset themselves.

        Args:
            env: The gymnasium environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, no clipping.
        """
        # imported lazily: isaaclab.envs only exists after SimulationApp starts
        from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. "
                f"Environment type: {type(env)}"
            )

        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by the runners
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runners do not call reset
        obs_dict, _ = self.env.reset()
        # observation group dims, e.g. {"policy": 35, "critic": 78}
        self.num_observations = {key: int(np.prod(value.shape[1:])) for key, value in obs_dict.items()}
        # privileged observation dim from the environment configuration
        self.num_privileged_obs = self._resolve_privileged_obs_dim()

    def _resolve_privileged_obs_dim(self) -> int | None:
        """Resolve the privileged (critic) observation dimension from the env cfg."""
        state_space = getattr(self.unwrapped.cfg, "state_space", None)
        if isinstance(state_space, dict):
            dim = state_space.get("critic", None)
            return int(dim) if isinstance(dim, int) and dim > 0 else None
        if isinstance(state_space, int) and state_space > 0:
            return state_space
        return None

    def __str__(self) -> str:
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self) -> str:
        return str(self)

    """
    Properties
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def unwrapped(self):
        """Returns the base environment underneath all wrappers."""
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer (needed for random episode-length init)."""
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        # reset the environment
        obs_dict, extras = self.env.reset()
        return self._as_obs_dict(obs_dict), extras

    def get_observations(self) -> dict[str, torch.Tensor]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return self._as_obs_dict(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # return the step information
        return self._as_obs_dict(obs_dict), rew, dones, extras

    def close(self):
        return self.env.close()

    """
    Helper functions
    """

    @staticmethod
    def _as_obs_dict(obs_dict) -> dict[str, torch.Tensor]:
        """Normalize an observation dict to a plain dict of tensors.

        Isaac Lab returns plain dicts while some wrappers return TensorDicts;
        the custom runners and storages expect plain dicts.
        """
        if isinstance(obs_dict, TensorDict):
            return {key: value for key, value in obs_dict.items()}
        return dict(obs_dict)

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
