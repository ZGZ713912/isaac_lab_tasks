"""Flattened-history MLP baseline with the suspension exploration floor."""

import math

from torch.distributions import Normal
from rsl_rl.modules import ActorCritic


class ActorCriticSuspensionMLP(ActorCritic):
    def __init__(
        self, obs, obs_groups, num_actions,
        actor_obs_normalization=False, critic_obs_normalization=False,
        actor_hidden_dims=(256, 256, 256), critic_hidden_dims=(256, 256, 256),
        activation="elu", init_noise_std=1.0, noise_std_type="scalar",
        history_length=1, min_noise_std=0.0, **kwargs,
    ):
        # These fields remain in the shared Transformer Hydra configuration.
        for key in ("d_model", "nhead", "num_layers", "dim_ff", "head_hidden",
                    "actor_head", "actor_layout", "critic_layout"):
            kwargs.pop(key, None)
        if kwargs.pop("state_dependent_std", False):
            raise ValueError("ActorCriticSuspensionMLP does not support state_dependent_std")
        if not math.isfinite(init_noise_std) or init_noise_std <= 0:
            raise ValueError("init_noise_std must be positive and finite")
        if not math.isfinite(min_noise_std) or min_noise_std < 0:
            raise ValueError("min_noise_std must be nonnegative and finite")
        width = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        if history_length not in (1, 4, 8) or width % history_length:
            raise ValueError("Policy observations must contain complete H=1, 4 or 8 frames")
        self.history_length = history_length
        self.min_noise_std = min_noise_std
        super().__init__(
            obs, obs_groups, num_actions, actor_obs_normalization, critic_obs_normalization,
            actor_hidden_dims, critic_hidden_dims, activation, init_noise_std, noise_std_type,
            **kwargs,
        )

    def update_distribution(self, obs):
        mean = self.actor(obs)
        std = self.log_std.exp() if self.noise_std_type == "log" else self.std
        self.distribution = Normal(mean, std.clamp_min(max(self.min_noise_std, 1.0e-6)).expand_as(mean))
