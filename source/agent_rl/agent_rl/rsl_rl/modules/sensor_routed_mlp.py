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
                 routing_confidence=.995, routing_load_threshold=.02, trainable_expert_indices=None):
        super().__init__()
        if (len(expert_hidden_dims) < 2 or not math.isfinite(routing_confidence)
                or not .5 < routing_confidence < 1 or not math.isfinite(routing_load_threshold)
                or routing_load_threshold < 0):
            raise ValueError("Routed MLP requires experts, finite confidence and a nonnegative load threshold")
        self.experts = nn.ModuleList([elu_mlp(160, dims, 4) for dims in expert_hidden_dims])
        if trainable_expert_indices is not None:
            if (not trainable_expert_indices or len(set(trainable_expert_indices)) != len(trainable_expert_indices)
                    or any(type(index) is not int or not 0 <= index < len(self.experts)
                           for index in trainable_expert_indices)):
                raise ValueError("Trainable experts must be distinct valid expert indices")
            for index,expert in enumerate(self.experts):
                expert.requires_grad_(index in trainable_expert_indices)
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


class SensorDefaultSplitMLP(SensorRoutedMLP):
    """Preserve the three-way router and specialize only its default branch.

    A fourth MLP is initialized from expert zero during explicit checkpoint
    assembly. A second frozen sensor MLP may select it only when the original
    routing result is zero, both confidence and load gates are satisfied, and
    its predicted class is one. Existing nondefault decisions take precedence.
    """

    def __init__(self, expert_hidden_dims, router_hidden_dims=(128, 64),
                 routing_confidence=.995, routing_load_threshold=.02,
                 trainable_expert_indices=None, split_router_hidden_dims=(128, 64),
                 split_routing_confidence=.995, moving_expert_hidden_dims=None,
                 standing_command_tolerance=1.e-6):
        # Exact cloning is checked by checkpoint assembly. A trained expert
        # may include a folded additive MLP with a wider hidden architecture.
        # Runtime construction must retain the saved architecture verbatim.
        if (len(expert_hidden_dims) != 4
                or not math.isfinite(split_routing_confidence) or not .5 < split_routing_confidence < 1):
            raise ValueError("Default split requires four experts and finite confidence")
        super().__init__(expert_hidden_dims, router_hidden_dims, routing_confidence,
                         routing_load_threshold, trainable_expert_indices)
        self.router = elu_mlp(160, router_hidden_dims, 3).requires_grad_(False)
        self.split_router = elu_mlp(160, split_router_hidden_dims, 2).requires_grad_(False)
        self.register_buffer("split_routing_confidence", torch.tensor(split_routing_confidence))
        if (not math.isfinite(standing_command_tolerance)
                or not 0 <= standing_command_tolerance <= 1.e-6):
            raise ValueError("Standing command tolerance must denote numerical zero")
        self.moving_expert = (elu_mlp(160,moving_expert_hidden_dims,4).requires_grad_(False)
                              if moving_expert_hidden_dims is not None else None)
        # Legacy checkpoints have no extra state. The command gate is saved
        # exactly when its frozen original grade17 branch is present.
        self.register_buffer("standing_command_tolerance",torch.tensor(standing_command_tolerance),
                             persistent=self.moving_expert is not None)


    def routing_indices(self, observations):
        confidence, route = self.router(observations).softmax(-1).max(-1)
        loaded = observations[:, -14:-10].abs().mean(-1) > self.routing_load_threshold
        route = torch.where((confidence >= self.routing_confidence) & loaded, route, torch.zeros_like(route))
        split_confidence, split_route = self.split_router(observations).softmax(-1).max(-1)
        select = ((route == 0) & loaded & (split_confidence >= self.split_routing_confidence) & (split_route == 1))
        if self.moving_expert is not None:
            standing=observations.reshape(-1,5,32)[:,:,1:4].abs().amax(dim=(1,2))<=self.standing_command_tolerance
            select=select & standing
        return torch.where(select, torch.full_like(route, 3), route)

    def forward(self, observations):
        route = self.routing_indices(observations)
        outputs = torch.stack([expert(observations) for expert in self.experts], dim=1)
        selected=outputs.gather(1,route[:,None,None].expand(-1,1,4)).squeeze(1)
        if self.moving_expert is not None:
            moving=observations.reshape(-1,5,32)[:,:,1:4].abs().amax(dim=(1,2))>self.standing_command_tolerance
            selected=torch.where((moving & (route==2))[:,None],self.moving_expert(observations),selected)
        return selected
