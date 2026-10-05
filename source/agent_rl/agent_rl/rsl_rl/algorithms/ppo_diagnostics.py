# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Repository-local diagnostics for the feed-forward rsl_rl 3.0.1 PPO update."""

import copy
import math
import torch
from rsl_rl.algorithms import PPO


class DiagnosticPPO(PPO):
    """Plain PPO with pre-clip gradient and rollout diagnostics.

    Metrics are returned alongside losses for the stock OnPolicyRunner logger.
    Actor gradients include the action-noise parameters. Explained variance uses
    stored rollout values, not repeatedly sampled minibatch predictions.
    """

    def __init__(self, policy, separate_grad_clip=False, steep_preservation_weight=0.0,
                 steep_reference_start_deg=3.0, steep_reference_full_deg=8.0,
                 steep_reference_action_scale=0.03, flat_posture_weight=0.0,
                 flat_posture_target_deg=19.0, flat_posture_tilt_deg=3.0,
                 flat_posture_anchor="mean", flat_posture_max_spread_deg=0.0,
                 reference_all_postures=False, **kwargs):
        if policy.is_recurrent or kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
            raise ValueError("DiagnosticPPO supports feed-forward PPO without RND or symmetry only")
        super().__init__(policy, **kwargs)
        self.separate_grad_clip = separate_grad_clip
        if steep_preservation_weight < 0 or not 0 <= steep_reference_start_deg < steep_reference_full_deg:
            raise ValueError("Invalid steep policy preservation weight or tilt interval")
        if steep_reference_action_scale <= 0:
            raise ValueError("Steep reference action scale must be positive")
        self.steep_preservation_weight = steep_preservation_weight
        self.steep_reference_start_deg = steep_reference_start_deg
        self.steep_reference_full_deg = steep_reference_full_deg
        self.steep_reference_action_scale = steep_reference_action_scale
        self.reference_all_postures = reference_all_postures
        self._steep_reference_actor = None
        self._steep_grade_reference_actor = None
        if flat_posture_weight < 0 or not 17.0 <= flat_posture_target_deg <= 75.0 or flat_posture_tilt_deg <= 0:
            raise ValueError("Invalid native V3 flat posture preference")
        self.flat_posture_weight = flat_posture_weight
        self.flat_posture_target_deg = flat_posture_target_deg
        self.flat_posture_tilt_deg = flat_posture_tilt_deg
        if flat_posture_anchor not in ("mean", "lowest"):
            raise ValueError("Flat posture anchor must be mean or lowest")
        self.flat_posture_anchor = flat_posture_anchor
        if not math.isfinite(flat_posture_max_spread_deg) or flat_posture_max_spread_deg < 0:
            raise ValueError("Flat posture leg spread must be finite and nonnegative")
        self.flat_posture_max_spread_deg = flat_posture_max_spread_deg
        self._critic_parameters = tuple(policy.critic.parameters())
        critic_ids = {id(parameter) for parameter in self._critic_parameters}
        # Includes std/log_std and any actor encoder, excluding critic parameters.
        self._actor_parameters = tuple(
            parameter for parameter in policy.parameters() if id(parameter) not in critic_ids
        )

    def initialize_steep_reference(self):
        """Freeze the loaded actor for training only; inference has no dependency."""
        if self.policy.actor_obs_normalization or getattr(self.policy.actor, "frame_size", None) != 32:
            raise ValueError("Steep preservation requires the unnormalized 32D deformable actor layout")
        self._steep_reference_actor = copy.deepcopy(self.policy.actor).eval().requires_grad_(False)

    def initialize_steep_grade_reference(self, actor_state):
        """Freeze a compatible second actor; its selector is training privilege."""
        if self._steep_reference_actor is None:
            raise RuntimeError("Initialize the primary steep reference first")
        actor = copy.deepcopy(self.policy.actor)
        actor.load_state_dict(actor_state, strict=True)
        self._steep_grade_reference_actor = actor.eval().requires_grad_(False)

    def _steep_preservation_loss(self, obs, action_mean):
        if self._steep_reference_actor is None:
            raise RuntimeError("Initialize the steep reference after loading fine-tune weights")
        actor_obs = self.policy.get_actor_obs(obs)
        gravity = actor_obs.reshape(actor_obs.shape[0], -1, 32)[:, -1, 7:10]
        tilt_deg = torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2]) * (180.0 / math.pi)
        gate = ((tilt_deg - self.steep_reference_start_deg)
                / (self.steep_reference_full_deg - self.steep_reference_start_deg)).clamp(0., 1.)
        if self.reference_all_postures:
            gate = torch.ones_like(gate)
        with torch.no_grad():
            target = self._steep_reference_actor(actor_obs)
            if self._steep_grade_reference_actor is not None:
                if "training_steep_reference_mix" not in obs:
                    raise ValueError("Steep grade reference requires the privileged training selector")
                mix = obs["training_steep_reference_mix"].clamp(0., 1.)
                target = target.lerp(self._steep_grade_reference_actor(actor_obs), mix)
        cost = ((action_mean - target) / self.steep_reference_action_scale).square().mean(-1)
        return (cost * gate).mean(), gate.mean()

    def _flat_posture_loss(self, obs, action_mean):
        """Soft common-height preference for the native 16/17/75 deg action map."""
        actor_obs = self.policy.get_actor_obs(obs)
        latest = actor_obs.reshape(actor_obs.shape[0], -1, 32)[:, -1]
        gravity = latest[:, 7:10]
        tilt_deg = torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2]) * (180.0 / math.pi)
        gate = (1.0 - tilt_deg / self.flat_posture_tilt_deg).clamp(0., 1.)
        if self.flat_posture_max_spread_deg:
            # Encoder angles are unscaled radians in current_fraction_v2.
            # A level body can still need unequal legs on sloping terrain.
            leg_angles = latest[:, 10:14]
            spread_deg = (leg_angles.amax(-1) - leg_angles.amin(-1)) * (180.0 / math.pi)
            gate = gate * (1.0 - spread_deg / self.flat_posture_max_spread_deg).clamp(0., 1.)
        action = action_mean.clamp(-1., 1.)
        # Positive actions cover one degree down; negative actions cover
        # 58 degrees up. This must match minangle_physical_v3, not legacy V2.
        physical_targets = 17.0 - action * torch.where(action < 0., 58.0, 1.0)
        angle = physical_targets.amin(-1) if self.flat_posture_anchor == "lowest" else physical_targets.mean(-1)
        excess = (angle - self.flat_posture_target_deg).clamp_min(0.)
        return (gate * (excess / 10.0).square()).mean(), gate.mean()

    @staticmethod
    def _grad_norm(parameters):
        norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
        return torch.stack(norms).norm(2) if norms else torch.tensor(0.0, device=parameters[0].device)

    def update(self):
        # Population moments allow global explained variance on distributed runs.
        with torch.no_grad():
            returns = self.storage.returns.flatten().double()
            residuals = returns - self.storage.values.flatten().double()
            moments = torch.stack(
                (returns.new_tensor(returns.numel()), returns.sum(), returns.square().sum(),
                 residuals.sum(), residuals.square().sum())
            )
            if self.is_multi_gpu:
                torch.distributed.all_reduce(moments)
            count, total, squares, residual_total, residual_squares = moments
            variance = (squares / count - (total / count).square()).clamp_min(0)
            residual_variance = (residual_squares / count - (residual_total / count).square()).clamp_min(0)
            explained_variance = (
                (1 - residual_variance / variance).item() if variance > 0 else float("nan")
            )

        metrics = {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "actor_grad_norm_pre_clip": 0.0,
            "critic_grad_norm_pre_clip": 0.0,
            "approx_kl": 0.0,
        }
        if self.steep_preservation_weight:
            metrics.update(steep_preservation=0.0, steep_reference_fraction=0.0)
            if self._steep_grade_reference_actor is not None:
                metrics["steep_grade_reference_fraction"] = 0.0
        if self.flat_posture_weight:
            metrics.update(flat_posture=0.0, flat_posture_fraction=0.0)
        mean_std = None
        num_updates = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch,
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, _, _,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.policy.act(obs_batch)
            log_prob = self.policy.get_actions_log_prob(actions_batch)
            values = self.policy.evaluate(obs_batch)
            mu = self.policy.action_mean
            sigma = self.policy.action_std
            entropy = self.policy.entropy.mean()

            with torch.no_grad():
                log_ratio = log_prob - old_actions_log_prob_batch.squeeze(-1)
                # Nonnegative sampled KL estimator: E[(ratio - 1) - log(ratio)].
                metrics["approx_kl"] += (log_ratio.expm1() - log_ratio).mean().item()
                batch_std = sigma.mean(dim=0).detach()
                mean_std = batch_std.clone() if mean_std is None else mean_std + batch_std

                if self.desired_kl is not None and self.schedule == "adaptive":
                    kl = (
                        torch.log(sigma / old_sigma_batch + 1.0e-5)
                        + (old_sigma_batch.square() + (old_mu_batch - mu).square()) / (2.0 * sigma.square())
                        - 0.5
                    ).sum(dim=-1).mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl)
                        kl /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl < self.desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr, src=0)
                        self.learning_rate = lr.item()
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(log_prob - old_actions_log_prob_batch.squeeze(-1))
            advantages = advantages_batch.squeeze(-1)
            surrogate = -advantages * ratio
            clipped_surrogate = -advantages * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.maximum(surrogate, clipped_surrogate).mean()
            if self.use_clipped_value_loss:
                clipped_values = target_values_batch + (values - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (values - returns_batch).square(), (clipped_values - returns_batch).square()
                ).mean()
            else:
                value_loss = (returns_batch - values).square().mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            if self.steep_preservation_weight:
                preservation_loss, reference_fraction = self._steep_preservation_loss(obs_batch, mu)
                loss = loss + self.steep_preservation_weight * preservation_loss
                metrics["steep_preservation"] += preservation_loss.item()
                metrics["steep_reference_fraction"] += reference_fraction.item()
                if self._steep_grade_reference_actor is not None:
                    metrics["steep_grade_reference_fraction"] += obs_batch["training_steep_reference_mix"].mean().item()
            if self.flat_posture_weight:
                flat_loss, flat_fraction = self._flat_posture_loss(obs_batch, mu)
                loss = loss + self.flat_posture_weight * flat_loss
                metrics["flat_posture"] += flat_loss.item()
                metrics["flat_posture_fraction"] += flat_fraction.item()
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            # Capture the actual weighted-loss gradients before upstream's global clip.
            metrics["actor_grad_norm_pre_clip"] += self._grad_norm(self._actor_parameters).item()
            metrics["critic_grad_norm_pre_clip"] += self._grad_norm(self._critic_parameters).item()
            if self.separate_grad_clip:
                torch.nn.utils.clip_grad_norm_(self._actor_parameters, self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self._critic_parameters, self.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            metrics["value_function"] += value_loss.item()
            metrics["surrogate"] += surrogate_loss.item()
            metrics["entropy"] += entropy.item()
            num_updates += 1

        metrics = {key: value / num_updates for key, value in metrics.items()}
        for index, std in enumerate(mean_std / num_updates):
            metrics[f"action_std_{index}"] = std.item()
        if self.is_multi_gpu:
            # Norms are already synchronized; average all other diagnostics/losses.
            reduced = torch.tensor(list(metrics.values()), dtype=torch.float64, device=self.device)
            torch.distributed.all_reduce(reduced)
            metrics = dict(zip(metrics, (reduced / self.gpu_world_size).tolist()))
        metrics["explained_variance"] = explained_variance
        self.storage.clear()
        return metrics
