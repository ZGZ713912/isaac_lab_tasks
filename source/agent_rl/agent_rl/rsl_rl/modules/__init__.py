from .actor_critic_ext import ActorCriticExt
from .actor_critic_dreamwaq import ActorCriticDreamWaq
from .actor_critic_him import ActorCriticHIM
from .actor_critic_balowtwins import ActorCriticBarlowTwins
from .actor_critic_transformer import ActorCriticTransformer
from .actor_critic_suspension_mlp import ActorCriticSuspensionMLP
from .actor_critic_suspension_routed_mlp import ActorCriticSuspensionRoutedMLP

# 原生 OnPolicyRunner 在自身模块命名空间里 eval(class_name)，注入后可直接
# 用 class_name="ActorCriticTransformer" 而无需自定义 runner。
import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner  # noqa: E402

_rsl_on_policy_runner.ActorCriticTransformer = ActorCriticTransformer
_rsl_on_policy_runner.ActorCriticSuspensionMLP = ActorCriticSuspensionMLP
_rsl_on_policy_runner.ActorCriticSuspensionRoutedMLP = ActorCriticSuspensionRoutedMLP

__all__ = [
    "ActorCriticSuspensionMLP",
    "ActorCriticSuspensionRoutedMLP",
    "ActorCriticTransformer",
    "ActorCriticExt",
    "ActorCriticDreamWaq",
    "ActorCriticHIM",
    "ActorCriticBarlowTwins",
]
