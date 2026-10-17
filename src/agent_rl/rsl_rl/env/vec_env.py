# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# =============================================================================

"""Abstract vectorized environment interface for RSL-RL style runners.

Mirrors ``rsl_rl.env.VecEnv`` (rsl-rl-lib) so that the custom runners
(DreamWaQ / HIM / NP3O) only depend on this package.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class VecEnv(ABC):
    """Abstract class for a vectorized environment.

    The vectorized environment is a collection of environments that are
    synchronized: the same action tensor is applied to all environments and
    the same observation structure is returned from all environments.
    """

    num_envs: int
    """Number of environments."""

    num_actions: int
    """Number of actions."""

    max_episode_length: int | torch.Tensor
    """Maximum episode length (scalar or per-environment tensor)."""

    episode_length_buf: torch.Tensor
    """Buffer for current episode lengths."""

    device: torch.device
    """Device on which the environment tensors live."""

    cfg: dict | object
    """Environment configuration object."""

    num_observations: dict[str, int]
    """Observation group dimensions, e.g. ``{"policy": 35, "critic": 78}``."""

    num_privileged_obs: int | None
    """Privileged observation dimension, or ``None`` if not available."""

    @abstractmethod
    def get_observations(self) -> dict[str, torch.Tensor]:
        """Return the current observations."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset the environments and return initial observations.

        Returns:
            observations (dict): Observation groups keyed by group name.
            extras (dict): Extra information from the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """Apply actions and step the environments.

        Returns:
            observations (dict): Next observations keyed by group name.
            rewards (torch.Tensor): Rewards, shape (num_envs,).
            dones (torch.Tensor): Done flags, shape (num_envs,).
            extras (dict): Extra information (may contain ``time_outs`` and ``log``).
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the environments."""
        raise NotImplementedError
