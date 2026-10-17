"""H5 suspension PPO with an explicitly separated default specialist."""

from .actor_critic_suspension_mlp import ActorCriticSuspensionMLP
from .sensor_routed_mlp import SensorDefaultSplitMLP


class ActorCriticSuspensionDefaultSplitMLP(ActorCriticSuspensionMLP):
    def __init__(self, obs, obs_groups, num_actions,
                 expert_hidden_dims=((576,288,144),(512,256,128),(704,352,176),(576,288,144)),
                 router_hidden_dims=(128,64), routing_confidence=.995,
                 routing_load_threshold=.02, trainable_expert_indices=None,
                 split_router_hidden_dims=(128,64), split_routing_confidence=.995,
                 moving_expert_hidden_dims=None, standing_command_tolerance=1.e-6, **kwargs):
        if (kwargs.get("history_length",1) != 5 or num_actions != 4
                or kwargs.get("activation","elu") != "elu"
                or kwargs.get("actor_obs_normalization",False)
                or kwargs.get("critic_obs_normalization",False)):
            raise ValueError("Default split suspension MLP requires raw H5 sensors, ELU and four actions")
        super().__init__(obs,obs_groups,num_actions,**kwargs)
        self.actor=SensorDefaultSplitMLP(expert_hidden_dims,router_hidden_dims,routing_confidence,
            routing_load_threshold,trainable_expert_indices,split_router_hidden_dims,split_routing_confidence,
            moving_expert_hidden_dims,standing_command_tolerance)
