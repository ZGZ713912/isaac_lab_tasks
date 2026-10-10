"""Five-frame MLP experts sharing the existing suspension PPO/critic contract."""

from .actor_critic_suspension_mlp import ActorCriticSuspensionMLP
from .sensor_routed_mlp import SensorRoutedMLP


class ActorCriticSuspensionRoutedMLP(ActorCriticSuspensionMLP):
    def __init__(self, obs, obs_groups, num_actions,
                 expert_hidden_dims=((576, 288, 144), (512, 256, 128), (704, 352, 176)),
                 router_hidden_dims=(128, 64), routing_confidence=.995,
                 routing_load_threshold=.02, **kwargs):
        if (kwargs.get("history_length", 1) != 5 or num_actions != 4
                or kwargs.get("activation", "elu") != "elu"
                or kwargs.get("actor_obs_normalization", False)
                or kwargs.get("critic_obs_normalization", False)):
            raise ValueError("Routed suspension MLP requires raw H5/160D sensors, ELU and four actions")
        super().__init__(obs, obs_groups, num_actions, **kwargs)
        self.actor = SensorRoutedMLP(expert_hidden_dims, router_hidden_dims,
                                    routing_confidence, routing_load_threshold)
