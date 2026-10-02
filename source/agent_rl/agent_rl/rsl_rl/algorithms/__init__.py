from .ppo_dreamwaq import PPODreamWaq
from .ppo_him import PPOHIM
from .np3o import NP3O
from .ppo_diagnostics import DiagnosticPPO

# Match the repository's policy registration: stock runner evals class_name here.
import rsl_rl.runners.on_policy_runner as _rsl_on_policy_runner

_rsl_on_policy_runner.DiagnosticPPO = DiagnosticPPO

__all__ = ["PPODreamWaq", "PPOHIM", "NP3O", "DiagnosticPPO"]
