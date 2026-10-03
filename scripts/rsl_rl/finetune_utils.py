"""Explicit exploration initialization after loading fine-tune weights."""

import math

import torch


def reset_policy_noise(policy, std):
    """Change only a state-independent action-noise parameter, preserving weights."""
    floor = max(float(getattr(policy, "min_noise_std", 0.0)), 1.e-6)
    if not math.isfinite(std) or std < floor:
        raise ValueError(f"Fine-tune noise must be finite and at least {floor}")
    if getattr(policy, "state_dependent_std", False):
        raise ValueError("Fine-tune noise reset requires state-independent noise")
    kind = getattr(policy, "noise_std_type", None)
    parameter = getattr(policy, "log_std" if kind == "log" else "std", None)
    if kind not in ("log", "scalar") or not isinstance(parameter, torch.nn.Parameter):
        raise ValueError("Policy does not expose a supported action-noise parameter")
    with torch.no_grad():
        before = (parameter.exp() if kind == "log" else parameter).clamp_min(floor).mean().item()
        parameter.fill_(math.log(std) if kind == "log" else std)
    return before
