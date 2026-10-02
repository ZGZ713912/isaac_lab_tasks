# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Repository-local diagnostics for the feed-forward rsl_rl 3.0.1 PPO update."""

import torch
from rsl_rl.algorithms import PPO


class DiagnosticPPO(PPO):
    """Plain PPO with pre-clip gradient and rollout diagnostics.

    Metrics are returned alongside losses for the stock OnPolicyRunner logger.
    Actor gradients include the action-noise parameters. Explained variance uses
    stored rollout values, not repeatedly sampled minibatch predictions.
    """

    def __init__(self, policy, separate_grad_clip=False, **kwargs):
        if policy.is_recurrent or kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
            raise ValueError("DiagnosticPPO supports feed-forward PPO without RND or symmetry only")
        super().__init__(policy, **kwargs)
        self.separate_grad_clip = separate_grad_clip
        self._critic_parameters = tuple(policy.critic.parameters())
        critic_ids = {id(parameter) for parameter in self._critic_parameters}
        # Includes std/log_std and any actor encoder, excluding critic parameters.
        self._actor_parameters = tuple(
            parameter for parameter in policy.parameters() if id(parameter) not in critic_ids
        )

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
