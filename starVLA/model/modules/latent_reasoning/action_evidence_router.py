"""Action-conditioned sparse routing over Space2Act latent channels."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_ROUTER_CHANNELS = ("spatial", "future", "plan")


class _ChannelEvidenceRouter(nn.Module):
    """Produce gated evidence slots for one typed latent channel."""

    def __init__(
        self,
        hidden_dim: int,
        router_dim: int,
        num_heads: int,
        action_token_count: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.num_heads = int(num_heads)
        self.head_dim = router_dim // self.num_heads
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, router_dim)
        self.key_projection = nn.Linear(hidden_dim, router_dim)
        self.value_projection = nn.Linear(hidden_dim, router_dim)
        self.output_projection = nn.Linear(router_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(router_dim * 2, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, 1),
        )
        self.null_evidence = nn.Parameter(torch.zeros(1, action_token_count, hidden_dim))

        # Start from a neutral 0.5 gate so action gradients, rather than initialization,
        # decide which slots remain active.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self, action_query: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query = self.query_projection(self.query_norm(action_query))
        normalized_memory = self.memory_norm(memory)
        key = self.key_projection(normalized_memory)
        value = self.value_projection(normalized_memory)

        batch_size, query_count, _ = query.shape
        memory_count = key.shape[1]
        query_heads = query.reshape(batch_size, query_count, self.num_heads, self.head_dim)
        query_heads = query_heads.transpose(1, 2)
        key_heads = key.reshape(batch_size, memory_count, self.num_heads, self.head_dim)
        key_heads = key_heads.transpose(1, 2)
        value_heads = value.reshape(batch_size, memory_count, self.num_heads, self.head_dim)
        value_heads = value_heads.transpose(1, 2)
        logits = torch.matmul(query_heads, key_heads.transpose(-1, -2))
        logits = logits / math.sqrt(self.head_dim)
        attention_heads = torch.softmax(logits.float(), dim=-1).to(dtype=value.dtype)
        pooled = torch.matmul(attention_heads, value_heads).transpose(1, 2).contiguous()
        pooled = pooled.reshape(batch_size, query_count, -1)
        attention = attention_heads.float().mean(dim=1)
        evidence = self.output_projection(pooled)
        gate_logits = self.gate(torch.cat((query, pooled), dim=-1))
        gate = torch.sigmoid(gate_logits / self.temperature)
        slots = gate * evidence + (1.0 - gate) * self.null_evidence.to(evidence.dtype)

        probability = attention.clamp_min(torch.finfo(torch.float32).eps)
        entropy = -(probability * probability.log()).sum(dim=-1)
        if memory.shape[1] > 1:
            entropy = entropy / math.log(memory.shape[1])

        if attention.shape[1] > 1:
            normalized_attention = F.normalize(attention, dim=-1)
            similarity = torch.matmul(normalized_attention, normalized_attention.transpose(-1, -2))
            count = attention.shape[1]
            diversity = (similarity.sum(dim=(-1, -2)) - count) / (count * (count - 1))
            diversity = diversity.mean()
        else:
            diversity = attention.new_zeros((), dtype=torch.float32)

        return slots, {
            "budget_loss": gate.float().mean(),
            "entropy_loss": entropy.mean(),
            "diversity_loss": diversity,
            "gate_mean": gate.float().mean().detach(),
            "active_ratio": (gate.float() >= 0.5).float().mean().detach(),
            "attention_entropy": entropy.mean().detach(),
        }


class ActionGuidedEvidenceRouter(nn.Module):
    """Route S/F/P hidden states with state-conditioned Action Intent queries."""

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        action_token_count: int = 8,
        router_dim: int = 512,
        num_heads: int = 8,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or router_dim <= 0:
            raise ValueError("hidden_dim and router_dim must be positive")
        if action_token_count <= 0:
            raise ValueError("action_token_count must be positive")
        if num_heads <= 0 or router_dim % num_heads:
            raise ValueError("router_dim must be divisible by num_heads")
        if not math.isfinite(gate_temperature) or gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be finite and positive")

        self.action_token_count = int(action_token_count)
        self.state_dim = int(state_dim)
        self.action_query_norm = nn.LayerNorm(hidden_dim)
        self.state_projection = (
            nn.Linear(self.state_dim, hidden_dim) if self.state_dim > 0 else None
        )
        self.channel_routers = nn.ModuleDict(
            {
                channel: _ChannelEvidenceRouter(
                    hidden_dim=hidden_dim,
                    router_dim=router_dim,
                    num_heads=num_heads,
                    action_token_count=self.action_token_count,
                    temperature=gate_temperature,
                )
                for channel in _ROUTER_CHANNELS
            }
        )

    def _condition_query(
        self, action_hidden: torch.Tensor, state: torch.Tensor | None
    ) -> torch.Tensor:
        if action_hidden.ndim != 3 or action_hidden.shape[1] != self.action_token_count:
            raise ValueError(
                "action_hidden must be [B,action_token_count,H], got "
                f"{tuple(action_hidden.shape)}"
            )
        conditioned = action_hidden
        if state is not None:
            if self.state_projection is None:
                raise ValueError("Router received state but state_dim is disabled")
            if state.ndim == 3:
                latest_state = state[:, -1]
            elif state.ndim == 2:
                latest_state = state
            else:
                raise ValueError(f"state must be [B,D] or [B,T,D], got {tuple(state.shape)}")
            if latest_state.shape[-1] != self.state_dim:
                raise ValueError(
                    f"Expected state dimension {self.state_dim}, got {latest_state.shape[-1]}"
                )
            conditioned = conditioned + self.state_projection(latest_state).unsqueeze(1)
        return self.action_query_norm(conditioned)

    def forward(
        self,
        action_hidden: torch.Tensor,
        channel_hidden: dict[str, torch.Tensor],
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if set(channel_hidden) != set(_ROUTER_CHANNELS):
            raise ValueError(
                "Evidence routing requires exactly spatial/future/plan hidden states, got "
                f"{sorted(channel_hidden)}"
            )
        action_query = self._condition_query(action_hidden, state)
        slots = []
        channel_stats = {}
        differentiable_stats = {name: [] for name in ("budget_loss", "entropy_loss", "diversity_loss")}
        for channel in _ROUTER_CHANNELS:
            memory = channel_hidden[channel]
            if memory.ndim != 3 or memory.shape[0] != action_hidden.shape[0]:
                raise ValueError(f"Invalid hidden shape for channel '{channel}': {tuple(memory.shape)}")
            channel_slots, stats = self.channel_routers[channel](action_query, memory)
            slots.append(channel_slots)
            for name in differentiable_stats:
                differentiable_stats[name].append(stats[name])
            for name in ("gate_mean", "active_ratio", "attention_entropy"):
                channel_stats[f"{name}/{channel}"] = stats[name]

        losses = {
            name: torch.stack(values).mean() for name, values in differentiable_stats.items()
        }
        losses.update(channel_stats)
        return torch.cat(slots, dim=1), losses


__all__ = ["ActionGuidedEvidenceRouter"]
