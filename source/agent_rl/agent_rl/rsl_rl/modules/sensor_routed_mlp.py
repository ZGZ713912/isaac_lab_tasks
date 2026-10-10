"""Stateless sensor-history MLP experts with a learned, frozen MLP router."""

import math
import torch
from torch import nn


def elu_mlp(inputs, hidden, outputs):
    if not hidden or any(type(width) is not int or width <= 0 for width in hidden):
        raise ValueError("MLP hidden dimensions must be positive integers")
    widths = [inputs, *hidden, outputs]
    layers = []
    for index, (before, after) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(before, after))
        if index < len(widths)-2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class SensorRoutedMLP(nn.Module):
    """Every branch and the router see only the same five raw sensor frames.

    Uncertain routing and unloaded startup retain expert zero. The selected
    branch's raw mean is returned directly, preserving its saturation feedback.
    No terrain label, privileged critic value or external controller is used.
    """

    def __init__(self, expert_hidden_dims, router_hidden_dims=(128, 64),
                 routing_confidence=.995, routing_load_threshold=.02):
        super().__init__()
        if (len(expert_hidden_dims) < 2 or not math.isfinite(routing_confidence)
                or not .5 < routing_confidence < 1 or not math.isfinite(routing_load_threshold)
                or routing_load_threshold < 0):
            raise ValueError("Routed MLP requires experts, finite confidence and a nonnegative load threshold")
        self.experts = nn.ModuleList([elu_mlp(160, dims, 4) for dims in expert_hidden_dims])
        self.router = elu_mlp(160, router_hidden_dims, len(expert_hidden_dims)).requires_grad_(False)
        self.register_buffer("routing_confidence", torch.tensor(routing_confidence))
        self.register_buffer("routing_load_threshold", torch.tensor(routing_load_threshold))
        self.frame_size = 32
        self.history_length = 5

    def forward(self, observations):
        confidence, route = self.router(observations).softmax(-1).max(-1)
        loaded = observations[:, -14:-10].abs().mean(-1) > self.routing_load_threshold
        route = torch.where((confidence >= self.routing_confidence) & loaded, route, torch.zeros_like(route))
        outputs = torch.stack([expert(observations) for expert in self.experts], dim=1)
        return outputs.gather(1, route[:, None, None].expand(-1, 1, 4)).squeeze(1)
