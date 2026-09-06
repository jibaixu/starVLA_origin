"""Causal latent dynamics bottleneck for joint Space2Act training.

The module operates only on student S/P/F hidden states.  Teacher features
remain targets for the existing alignment heads and are never consumed by the
dynamics predictor, so the predictor can be removed completely at inference.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class CausalDynamicsBottleneck(nn.Module):
    """Learn an explicit S/P -> F transition in a shared compact latent space.

    ``mode='residual'`` computes only the forward residual transition.  The
    default ``hybrid`` mode adds an inverse structural equation and a batch
    intervention ranking loss which prevents the forward head from ignoring P.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        dynamics_dim: int = 256,
        mode: str = "hybrid",
        margin: float = 0.1,
        variance_floor: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or state_dim < 0 or dynamics_dim <= 0:
            raise ValueError("hidden_dim and dynamics_dim must be positive; state_dim must be non-negative")
        mode = str(mode).lower()
        if mode not in {"residual", "structural", "hybrid"}:
            raise ValueError("dynamics bottleneck mode must be residual, structural, or hybrid")
        if margin < 0.0 or not torch.isfinite(torch.tensor(float(margin))):
            raise ValueError("dynamics bottleneck margin must be finite and non-negative")
        if variance_floor < 0.0 or not torch.isfinite(torch.tensor(float(variance_floor))):
            raise ValueError("dynamics bottleneck variance_floor must be finite and non-negative")

        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.dynamics_dim = int(dynamics_dim)
        self.mode = mode
        self.margin = float(margin)
        self.variance_floor = float(variance_floor)

        self.pool_key = nn.Linear(hidden_dim, dynamics_dim, bias=False)
        self.pool_value = nn.Linear(hidden_dim, dynamics_dim)
        self.pool_queries = nn.Parameter(torch.empty(3, dynamics_dim))
        nn.init.normal_(self.pool_queries, mean=0.0, std=dynamics_dim**-0.5)
        self.channel_norm = nn.ModuleDict(
            {channel: nn.LayerNorm(dynamics_dim) for channel in ("spatial", "plan", "future")}
        )

        condition_dim = dynamics_dim * 2 + state_dim
        self.forward_delta = _mlp(condition_dim, dynamics_dim * 2, dynamics_dim)
        self.forward_gate = _mlp(condition_dim, dynamics_dim, dynamics_dim)
        self.inverse_plan = _mlp(condition_dim, dynamics_dim * 2, dynamics_dim)

    def _pool(self, hidden: torch.Tensor, channel_index: int, channel: str) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim or hidden.shape[1] <= 0:
            raise ValueError(
                f"{channel} hidden must be [B,K,{self.hidden_dim}], got {tuple(hidden.shape)}"
            )
        # Keep the compact auxiliary head numerically stable when callers pass
        # BF16 hidden states outside a CUDA autocast context.
        hidden = hidden.to(device=self.pool_key.weight.device, dtype=self.pool_key.weight.dtype)
        key = self.pool_key(hidden)
        value = self.pool_value(hidden)
        query = self.pool_queries[channel_index].to(dtype=key.dtype, device=key.device)
        scores = torch.einsum("bkd,d->bk", key, query) / self.dynamics_dim**0.5
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bk,bkd->bd", weights, value)
        return self.channel_norm[channel](pooled.float()).to(dtype=hidden.dtype)

    def _state(self, state: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
        if self.state_dim == 0:
            return reference.new_zeros((reference.shape[0], 0))
        if state is None:
            return reference.new_zeros((reference.shape[0], self.state_dim))
        if state.ndim == 3:
            state = state[:, -1, :]
        if state.ndim != 2 or state.shape[0] != reference.shape[0] or state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must be [B,{self.state_dim}] or [B,T,{self.state_dim}], got {tuple(state.shape)}"
            )
        return state.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = F.layer_norm(prediction.float(), (prediction.shape[-1],))
        target = F.layer_norm(target.float(), (target.shape[-1],))
        cosine = 1.0 - F.cosine_similarity(prediction, target.detach(), dim=-1)
        mse = (prediction - target.detach()).square().mean(dim=-1)
        return (cosine + 0.1 * mse).mean()

    def forward(
        self,
        channel_hidden: Mapping[str, torch.Tensor],
        state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        expected = {"spatial", "plan", "future"}
        if set(channel_hidden) != expected:
            raise ValueError(
                "CausalDynamicsBottleneck requires exactly spatial/plan/future hidden states"
            )

        z_s = self._pool(channel_hidden["spatial"], 0, "spatial")
        z_p = self._pool(channel_hidden["plan"], 1, "plan")
        z_f = self._pool(channel_hidden["future"], 2, "future")
        q = self._state(state, z_s)
        condition = torch.cat((z_s, z_p, q), dim=-1)
        delta = self.forward_delta(condition)
        gate = torch.sigmoid(self.forward_gate(condition))
        predicted_future = F.layer_norm(z_s.float() + gate.float() * delta.float(), (self.dynamics_dim,))
        predicted_future = predicted_future.to(dtype=z_s.dtype)

        residual_loss = self._distance(predicted_future, z_f)
        zero = residual_loss.new_zeros(())
        structural_loss = zero
        intervention_loss = zero
        inverse_plan = z_p
        intervention_margin = zero
        if self.mode in {"structural", "hybrid"}:
            inverse_condition = torch.cat((z_s, z_f, q), dim=-1)
            inverse_plan = F.layer_norm(self.inverse_plan(inverse_condition).float(), (self.dynamics_dim,))
            inverse_plan = inverse_plan.to(dtype=z_p.dtype)
            structural_loss = self._distance(inverse_plan, z_p)

            if z_s.shape[0] > 1:
                permutation = torch.roll(torch.arange(z_s.shape[0], device=z_s.device), shifts=1)
                wrong_condition = torch.cat((z_s, z_p[permutation], q), dim=-1)
                wrong_delta = self.forward_delta(wrong_condition)
                wrong_gate = torch.sigmoid(self.forward_gate(wrong_condition))
                wrong_future = F.layer_norm(
                    z_s.float() + wrong_gate.float() * wrong_delta.float(), (self.dynamics_dim,)
                )
                wrong_future = wrong_future.to(dtype=z_s.dtype)
                positive = F.cosine_similarity(predicted_future.float(), z_f.float().detach(), dim=-1)
                negative = F.cosine_similarity(wrong_future.float(), z_f.float().detach(), dim=-1)
                intervention_values = F.relu(self.margin - positive + negative)
                intervention_loss = intervention_values.mean()
                intervention_margin = (positive - negative).mean()

        stacked = torch.stack((z_s.float(), z_p.float(), z_f.float()), dim=1)
        batch_std = stacked.std(dim=0, unbiased=False)
        variance_loss = F.relu(self.variance_floor - batch_std).mean()
        total = residual_loss + structural_loss + intervention_loss + variance_loss
        return {
            "loss": total,
            "residual_loss": residual_loss,
            "structural_loss": structural_loss,
            "intervention_loss": intervention_loss,
            "variance_loss": variance_loss,
            "intervention_margin": intervention_margin,
            "predicted_future": predicted_future,
            "inverse_plan": inverse_plan,
            "z_spatial": z_s,
            "z_plan": z_p,
            "z_future": z_f,
        }
