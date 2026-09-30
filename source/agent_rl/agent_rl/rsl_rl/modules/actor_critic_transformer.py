# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
# =============================================================================
"""单帧结构化 token 的 Transformer Actor-Critic（rsl_rl 3.x 接口兼容）。

把扁平观测按“全局 + 每腿”拆成 1 + L 个 token，用自注意力建模腿间耦合：
    global token : 车体级量（指令、角速度、重力投影；critic 另含特权量）
    leg token i  : 第 i 条腿的量（pos/vel/torque/act；critic 另含轮接触力）
腿 token 共享 embedding，另加可学习腿身份编码以区分四条腿。

输出：
    actor  : head="per_leg" —— 每个腿 token 经共享头输出该腿 1 维动作（要求 L == num_actions）
             head="global"  —— 全局 token 经头输出 num_actions 维
    critic : 全局 token 经头输出价值

注意力手写实现（不用 nn.MultiheadAttention），避免 eval 快路径对 ONNX/TorchScript 的影响。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization


# deformable_suspension 的默认观测合同（见 env_cfg.py 头注释）
#   policy 26 = q_cmd1|cmd3|angvel3|grav3|leg_pos4|leg_vel4|leg_torque4|act4
#   critic 34 = policy26 | lin_vel3 | height1 | wheel_force4
DEFORMABLE_ACTOR_LAYOUT = {
    "global": list(range(0, 10)),
    "legs": [[10 + i, 14 + i, 18 + i, 22 + i] for i in range(4)],
}
DEFORMABLE_CRITIC_LAYOUT = {
    "global": list(range(0, 10)) + [26, 27, 28, 29],
    "legs": [[10 + i, 14 + i, 18 + i, 22 + i, 30 + i] for i in range(4)],
}


class _EncoderLayer(nn.Module):
    """Pre-LN Transformer 编码层（手写多头注意力）。"""

    def __init__(self, d_model: int, nhead: int, dim_ff: int):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, t, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (b, nhead, t, head_dim)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(out)
        x = x + self.ff(self.norm2(x))
        return x


class LegTokenTransformer(nn.Module):
    """扁平观测 → (1 + L) token → Transformer → 输出。"""

    def __init__(
        self,
        num_obs: int,
        num_outputs: int,
        layout: dict,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        head_hidden: int = 64,
        head: str = "global",
        out_gain: float = 1.0,
    ):
        super().__init__()
        global_idx = list(layout["global"])
        legs_idx = [list(l) for l in layout["legs"]]
        num_legs = len(legs_idx)
        leg_dim = len(legs_idx[0])
        assert all(len(l) == leg_dim for l in legs_idx), "all leg tokens must have the same size"
        max_idx = max(global_idx + [i for l in legs_idx for i in l])
        assert max_idx < num_obs, f"layout index {max_idx} out of range for obs dim {num_obs}"
        if head == "per_leg":
            assert num_outputs == num_legs, "per_leg head requires num_outputs == num_legs"

        self.num_obs = num_obs
        self.num_outputs = num_outputs
        self.head_type = head
        self.register_buffer("global_idx", torch.tensor(global_idx, dtype=torch.long), persistent=False)
        self.register_buffer(
            "legs_idx", torch.tensor(legs_idx, dtype=torch.long).reshape(-1), persistent=False
        )
        self.num_legs = num_legs
        self.leg_dim = leg_dim

        self.global_embed = nn.Linear(len(global_idx), d_model)
        self.leg_embed = nn.Linear(leg_dim, d_model)
        # token 身份编码：0=global，1..L=各腿
        self.token_pos = nn.Parameter(torch.zeros(1, 1 + num_legs, d_model))
        nn.init.normal_(self.token_pos, std=0.02)

        self.layers = nn.ModuleList([_EncoderLayer(d_model, nhead, dim_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

        head_out = 1 if head == "per_leg" else num_outputs
        self.head = nn.Sequential(nn.Linear(d_model, head_hidden), nn.ELU(), nn.Linear(head_hidden, head_out))
        nn.init.orthogonal_(self.head[-1].weight, gain=out_gain)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        g = self.global_embed(obs.index_select(-1, self.global_idx)).unsqueeze(1)  # (b,1,d)
        legs = obs.index_select(-1, self.legs_idx).reshape(b, self.num_legs, self.leg_dim)
        l = self.leg_embed(legs)  # (b,L,d)
        x = torch.cat([g, l], dim=1) + self.token_pos
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.head_type == "per_leg":
            return self.head(x[:, 1:, :]).squeeze(-1)  # (b,L)
        return self.head(x[:, 0, :])


class ActorCriticTransformer(ActorCritic):
    """rsl_rl ActorCritic 的 Transformer 版本；接口与原生 ActorCritic 一致。"""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        head_hidden: int = 64,
        actor_head: str = "per_leg",
        actor_layout: dict | None = None,
        critic_layout: dict | None = None,
        **kwargs,
    ):
        # RslRlPpoActorCriticCfg 基类自带字段，对 transformer 无意义
        for _k in ("actor_hidden_dims", "critic_hidden_dims", "activation"):
            kwargs.pop(_k, None)
        if kwargs.pop("state_dependent_std", False):
            raise ValueError("ActorCriticTransformer does not support state_dependent_std")
        if kwargs:
            print(
                "ActorCriticTransformer.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        nn.Module.__init__(self)

        self.obs_groups = obs_groups
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        actor_layout = actor_layout or DEFORMABLE_ACTOR_LAYOUT
        critic_layout = critic_layout or DEFORMABLE_CRITIC_LAYOUT

        common = dict(d_model=d_model, nhead=nhead, num_layers=num_layers, dim_ff=dim_ff, head_hidden=head_hidden)
        self.actor = LegTokenTransformer(
            num_actor_obs, num_actions, actor_layout, head=actor_head, out_gain=0.01, **common
        )
        self.critic = LegTokenTransformer(num_critic_obs, 1, critic_layout, head="global", out_gain=1.0, **common)

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else nn.Identity()
        )
        n_a = sum(p.numel() for p in self.actor.parameters())
        n_c = sum(p.numel() for p in self.critic.parameters())
        print(f"Actor Transformer ({n_a} params): {self.actor}")
        print(f"Critic Transformer ({n_c} params)")

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)
