"""VLM-only Space2Act teacher-distillation framework.

This stage deliberately has no action head.  It learns the Qwen backbone,
typed S/P/F token embeddings, and alignment mappers from cached teachers; the
resulting ``qwen_vl_interface`` is later loaded and frozen by the independent
Space2Act action-head framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.space2act_two_stage import (
    SPACE2ACT_CHANNELS,
    interpolate_loss_schedule,
    is_enabled,
    normalize_confidence_aware_config,
    normalize_gradnorm_config,
    normalize_loss_schedule,
    project_bounded_simplex,
)
from starVLA.model.modules.latent_reasoning import CrossAttentionLatentAligner
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _channel_defaults(target_tokens: int, target_dim: int) -> dict:
    return {
        "token_enabled": True,
        "token_count": 32,
        "alignment_enabled": True,
        "target_tokens": target_tokens,
        "target_dim": target_dim,
        "alignment_dim": 1024,
        "num_heads": 8,
        "loss_weight": 1.0,
        "mse_weight": 0.1,
    }


@dataclass
class QwenSpace2ActVLMDefaultConfig:
    """Only configuration that the VLM-only distillation stage consumes."""

    name: str = "QwenSpace2ActVLM"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
        }
    )
    latent_reasoning: dict = field(
        default_factory=lambda: {
            "routing_mode": "native_causal",
            "token_order": "spatial_instruction_plan_future",
            "token_init_strategy": "avg",
            "query_chunk_size": 128,
            "confidence_aware_alignment": {
                "enabled": False,
                "min_weight": 0.0,
                "require_cached_confidence": True,
            },
            # `enabled=false` is fixed.  `enabled=true` supports progressive
            # knots or VLM-only GradNorm; DWA is deliberately unsupported.
            "loss_schedule": {"enabled": False, "mode": "fixed", "knots": []},
            "channels": {
                "spatial": _channel_defaults(1521, 5120),
                "future": _channel_defaults(1125, 3072),
                "plan": _channel_defaults(32, 32),
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenSpace2ActVLM")
class QwenSpace2Act_VLM(baseframework):
    """Teacher-only pretraining of Space2Act latent reasoning tokens."""

    @staticmethod
    def _token_positions(input_ids: torch.Tensor, token_ids: list[int], label: str) -> torch.Tensor:
        rows = []
        for batch_index in range(input_ids.shape[0]):
            positions = []
            for token_id in token_ids:
                matches = torch.nonzero(input_ids[batch_index] == token_id, as_tuple=False).flatten()
                if matches.numel() != 1:
                    raise ValueError(
                        f"Expected exactly one '{label}' token id {token_id}, found {matches.numel()}"
                    )
                positions.append(matches[0])
            rows.append(torch.stack(positions))
        return torch.stack(rows)

    def _typed_hidden(self, input_ids: torch.Tensor, hidden: torch.Tensor, channel: str) -> torch.Tensor:
        positions = self._token_positions(
            input_ids, self.qwen_vl_interface.space2act_token_ids[channel], channel
        )
        return torch.gather(
            hidden, dim=1, index=positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        )

    def _confidence_weighted_alignment_loss(
        self, per_example_loss: torch.Tensor, confidence: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        confidence = confidence.to(device=per_example_loss.device, dtype=torch.float32).clamp(0.0, 1.0)
        weights = self.confidence_aware_config["min_weight"] + (
            1.0 - self.confidence_aware_config["min_weight"]
        ) * confidence
        return (per_example_loss.float() * weights).mean(), {
            "raw_loss": per_example_loss.float().mean(),
            "confidence_mean": confidence.mean(),
            "effective_weight_mean": weights.mean(),
        }

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenSpace2ActVLMDefaultConfig, config)
        latent_cfg = self.config.framework.latent_reasoning

        if str(latent_cfg.get("routing_mode", "native_causal")) != "native_causal":
            raise ValueError("QwenSpace2ActVLM only supports routing_mode=native_causal")
        schedule_cfg = latent_cfg.get("loss_schedule", {})
        schedule_mode = str(schedule_cfg.get("mode", "fixed")).lower()
        schedule_enabled = is_enabled(schedule_cfg.get("enabled", False))
        if schedule_mode not in {"fixed", "progressive", "gradnorm"}:
            raise ValueError(
                "QwenSpace2ActVLM loss_schedule.mode must be fixed, progressive, or gradnorm"
            )
        if schedule_mode == "fixed" and schedule_enabled:
            raise ValueError("VLM-only fixed loss_schedule requires enabled=false")
        if schedule_mode in {"progressive", "gradnorm"} and not schedule_enabled:
            raise ValueError(f"VLM-only {schedule_mode} loss_schedule requires enabled=true")

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        token_order = str(latent_cfg.get("token_order", "spatial_instruction_future_plan"))
        self.qwen_vl_interface.configure_space2act_tokens(
            latent_cfg.channels,
            init_strategy=str(latent_cfg.get("token_init_strategy", "avg")),
            action_token_count=0,
            token_order=token_order,
        )

        disabled = [
            channel for channel in SPACE2ACT_CHANNELS
            if not is_enabled(latent_cfg.channels[channel].get("token_enabled", False))
            or not is_enabled(latent_cfg.channels[channel].get("alignment_enabled", False))
        ]
        if disabled:
            raise ValueError(
                "QwenSpace2ActVLM requires token_enabled and alignment_enabled for all "
                f"S/P/F channels; invalid channels: {disabled}"
            )

        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        self.alignment_heads = nn.ModuleDict()
        self.alignment_loss_weights: dict[str, float] = {}
        for channel in SPACE2ACT_CHANNELS:
            channel_cfg = latent_cfg.channels[channel]
            self.alignment_heads[channel] = CrossAttentionLatentAligner(
                input_dim=hidden_dim,
                target_tokens=int(channel_cfg.target_tokens),
                target_dim=int(channel_cfg.target_dim),
                alignment_dim=int(channel_cfg.get("alignment_dim", 1024)),
                num_heads=int(channel_cfg.get("num_heads", 8)),
                query_chunk_size=int(latent_cfg.get("query_chunk_size", 128)),
                mse_weight=float(channel_cfg.get("mse_weight", 0.1)),
            )
            self.alignment_loss_weights[channel] = float(channel_cfg.get("loss_weight", 1.0))
        self.confidence_aware_config = normalize_confidence_aware_config(
            latent_cfg.get("confidence_aware_alignment", {})
        )
        self.loss_schedule_knots = normalize_loss_schedule(
            schedule_cfg if schedule_mode == "progressive" else {}, self.alignment_heads.keys()
        )
        self.progressive_enabled = bool(self.loss_schedule_knots)
        self.gradnorm_config = normalize_gradnorm_config(
            schedule_cfg if schedule_mode == "gradnorm" else {}, self.alignment_heads.keys()
        )
        self.gradnorm_enabled = bool(self.gradnorm_config["enabled"])
        self.gradnorm_channels = tuple(self.alignment_heads.keys()) if self.gradnorm_enabled else ()
        budget = float(self.gradnorm_config.get("auxiliary_budget", 0.0))
        initial_weight = budget / len(self.gradnorm_channels) if self.gradnorm_channels else 0.0
        self.gradnorm_weights = {channel: initial_weight for channel in self.gradnorm_channels}
        self.gradnorm_gradient_ema = {channel: None for channel in self.gradnorm_channels}
        self.gradnorm_last_gradient_norm = {channel: 0.0 for channel in self.gradnorm_channels}
        self.gradnorm_window_probes = 0

    def _gradnorm_ramp_factor(self, training_step: int) -> float:
        warmup_steps = int(self.gradnorm_config["warmup_steps"])
        if training_step <= warmup_steps:
            return 0.0
        ramp_steps = int(self.gradnorm_config.get("adaptive_weight_ramp_steps", 0))
        if ramp_steps <= 0:
            return 1.0
        progress = min(1.0, (training_step - warmup_steps) / ramp_steps)
        return 0.5 * (1.0 - math.cos(math.pi * progress))

    def _effective_alignment_weights(self, training_step: int | None) -> dict[str, float]:
        if self.gradnorm_enabled:
            if training_step is None:
                raise ValueError("training_step is required when VLM-only GradNorm is enabled")
            step = int(training_step)
            budget = float(self.gradnorm_config["auxiliary_budget"])
            uniform = budget / len(self.gradnorm_channels)
            ramp = self._gradnorm_ramp_factor(step)
            return {
                channel: (1.0 - ramp) * uniform + ramp * self.gradnorm_weights[channel]
                for channel in self.gradnorm_channels
            }
        if self.progressive_enabled:
            if training_step is None:
                raise ValueError("training_step is required when VLM-only progressive scheduling is enabled")
            return interpolate_loss_schedule(self.loss_schedule_knots, int(training_step))
        return dict(self.alignment_loss_weights)

    def _compute_gradnorm_probe(
        self, last_hidden: torch.Tensor, channel_losses: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        probes = {}
        for channel in self.gradnorm_channels:
            gradient = torch.autograd.grad(
                channel_losses[channel], last_hidden, retain_graph=True, create_graph=False
            )[0]
            probes[f"gradnorm_probe/{channel}"] = torch.linalg.vector_norm(gradient.detach().float())
        return probes

    def _gradnorm_balance_error(self) -> float:
        weighted = [
            self.gradnorm_weights[channel] * float(self.gradnorm_gradient_ema[channel])
            for channel in self.gradnorm_channels
            if self.gradnorm_gradient_ema[channel] is not None
        ]
        if len(weighted) != len(self.gradnorm_channels) or min(weighted, default=0.0) <= 1e-12:
            return 0.0
        return max(weighted) / min(weighted)

    def update_adaptive_state(
        self,
        training_step: int,
        channel_losses: dict[str, torch.Tensor] | None = None,
        should_update: bool = True,
        channel_gradient_norms: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Update VLM-only GradNorm after an optimizer step; fixed/progressive are stateless."""
        if not self.gradnorm_enabled:
            return {}
        gradient_norms = channel_gradient_norms or {}
        previous_weights = dict(self.gradnorm_weights)
        step = int(training_step)
        updated = False
        if should_update and step > int(self.gradnorm_config["warmup_steps"]):
            if gradient_norms:
                if any(channel not in gradient_norms for channel in self.gradnorm_channels):
                    raise ValueError("VLM-only GradNorm probe is missing an alignment channel")
                values = torch.stack(
                    [gradient_norms[channel].detach().float() for channel in self.gradnorm_channels]
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                    values.div_(dist.get_world_size())
                decay = float(self.gradnorm_config["gradient_ema_decay"])
                for index, channel in enumerate(self.gradnorm_channels):
                    value = float(values[index].item())
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(f"Invalid VLM-only GradNorm probe for '{channel}'")
                    previous = self.gradnorm_gradient_ema[channel]
                    self.gradnorm_gradient_ema[channel] = value if previous is None else decay * previous + (1.0 - decay) * value
                    self.gradnorm_last_gradient_norm[channel] = value
                self.gradnorm_window_probes += 1

            update_interval = int(self.gradnorm_config["gradient_update_interval"])
            if (step - int(self.gradnorm_config["warmup_steps"])) % update_interval == 0:
                expected_probes = update_interval // int(self.gradnorm_config["gradient_probe_interval"])
                if self.gradnorm_window_probes == expected_probes and all(
                    self.gradnorm_gradient_ema[channel] is not None for channel in self.gradnorm_channels
                ):
                    epsilon = float(self.gradnorm_config["gradient_epsilon"])
                    inverse_norms = {
                        channel: 1.0 / max(float(self.gradnorm_gradient_ema[channel]), epsilon)
                        for channel in self.gradnorm_channels
                    }
                    budget = float(self.gradnorm_config["auxiliary_budget"])
                    total_inverse = sum(inverse_norms.values())
                    targets = {channel: budget * inverse_norms[channel] / total_inverse for channel in self.gradnorm_channels}
                    self.gradnorm_weights = project_bounded_simplex(
                        targets,
                        total=budget,
                        lower=float(self.gradnorm_config["min_weight"]),
                        upper=float(self.gradnorm_config["max_weight"]),
                    )
                    updated = True
                self.gradnorm_window_probes = 0

        metrics = {
            "gradnorm_state_updated": float(updated),
            "gradnorm_weight_sum": float(sum(self.gradnorm_weights.values())),
            "gradnorm_ready": float(all(self.gradnorm_gradient_ema[channel] is not None for channel in self.gradnorm_channels)),
            "gradnorm_window_probes": float(self.gradnorm_window_probes),
            "gradnorm_balance_error": self._gradnorm_balance_error(),
            "gradnorm/ramp_factor": self._gradnorm_ramp_factor(step),
        }
        for channel in self.gradnorm_channels:
            ema = self.gradnorm_gradient_ema[channel]
            metrics[f"gradnorm/ema_gradient_norm/{channel}"] = float(ema or 0.0)
            metrics[f"gradnorm/raw_gradient_norm/{channel}"] = float(self.gradnorm_last_gradient_norm[channel])
            metrics[f"gradnorm/weight/{channel}"] = float(self.gradnorm_weights[channel])
            metrics[f"gradnorm/weight_delta/{channel}"] = float(self.gradnorm_weights[channel] - previous_weights[channel])
        return metrics

    def get_adaptive_state(self) -> dict:
        if not self.gradnorm_enabled:
            return {"mode": "fixed"}
        return {
            "mode": "gradnorm",
            "channels": list(self.gradnorm_channels),
            "weights": dict(self.gradnorm_weights),
            "gradient_ema": dict(self.gradnorm_gradient_ema),
            "last_gradient_norm": dict(self.gradnorm_last_gradient_norm),
            "window_probes": self.gradnorm_window_probes,
        }

    def set_adaptive_state(self, state: dict | None) -> None:
        if not self.gradnorm_enabled or not state:
            return
        if state.get("mode") != "gradnorm" or tuple(state.get("channels", ())) != self.gradnorm_channels:
            raise ValueError("VLM-only GradNorm checkpoint sidecar is incompatible with this run")
        for channel in self.gradnorm_channels:
            ema = state.get("gradient_ema", {}).get(channel)
            weight = float(state.get("weights", {}).get(channel, self.gradnorm_weights[channel]))
            if ema is not None and (not math.isfinite(float(ema)) or float(ema) < 0.0):
                raise ValueError(f"Invalid VLM-only GradNorm EMA for '{channel}'")
            if not math.isfinite(weight):
                raise ValueError(f"Invalid VLM-only GradNorm weight for '{channel}'")
            self.gradnorm_gradient_ema[channel] = None if ema is None else float(ema)
            self.gradnorm_last_gradient_norm[channel] = float(
                state.get("last_gradient_norm", {}).get(channel, 0.0)
            )
            self.gradnorm_weights[channel] = weight
        self.gradnorm_window_probes = int(state.get("window_probes", 0))

    def space2act_layout(self) -> dict:
        latent_cfg = self.config.framework.latent_reasoning
        return {
            "base_vlm": str(self.config.framework.qwenvl.base_vlm),
            "token_order": str(latent_cfg.get("token_order", "spatial_instruction_future_plan")),
            "channels": {
                channel: int(latent_cfg.channels[channel].token_count)
                for channel in SPACE2ACT_CHANNELS
            },
        }

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        """Compute only S/P/F teacher alignment losses; actions are intentionally ignored."""
        training_step = kwargs.pop("training_step", None)
        run_gradient_probe = kwargs.pop("run_gradient_probe", False)
        if not examples:
            raise ValueError("QwenSpace2ActVLM requires a non-empty batch of examples")
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        losses = {}
        total_alignment_loss = last_hidden.new_zeros((), dtype=torch.float32)
        channel_losses = {}
        effective_weights = self._effective_alignment_weights(training_step)
        for channel, aligner in self.alignment_heads.items():
            if any(
                "space2act_teacher" not in example
                or channel not in example["space2act_teacher"]
                for example in examples
            ):
                raise KeyError(f"Missing Space2Act teacher tensors for enabled channel '{channel}'")
            compact_hidden = self._typed_hidden(
                qwen_inputs["input_ids"], last_hidden, channel
            )
            teachers = [example["space2act_teacher"][channel] for example in examples]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.confidence_aware_config["enabled"]:
                    confidence_available = all(
                        "space2act_confidence" in example
                        and channel in example["space2act_confidence"]
                        for example in examples
                    )
                    if not confidence_available and self.confidence_aware_config["require_cached_confidence"]:
                        raise KeyError(
                            f"Missing Space2Act confidence tensors for enabled channel '{channel}'"
                        )
                    confidence = (
                        torch.stack(
                            [
                                torch.as_tensor(example["space2act_confidence"][channel]).reshape(())
                                for example in examples
                            ]
                        ).to(device=last_hidden.device, dtype=torch.float32, non_blocking=True)
                        if confidence_available
                        else torch.ones(len(examples), device=last_hidden.device, dtype=torch.float32)
                    )
                    per_example_loss = aligner.compute_per_example_loss(compact_hidden, teachers)
                    channel_loss, metrics = self._confidence_weighted_alignment_loss(
                        per_example_loss, confidence
                    )
                    losses[f"{channel}_alignment_raw_loss"] = metrics["raw_loss"]
                    losses[f"{channel}_confidence_mean"] = metrics["confidence_mean"]
                    losses[f"{channel}_effective_weight_mean"] = metrics["effective_weight_mean"]
                else:
                    channel_loss = aligner.compute_loss(compact_hidden, teachers)

            weight = self.alignment_loss_weights[channel]
            weight = effective_weights[channel]
            weighted_loss = weight * channel_loss
            losses[f"{channel}_alignment_weight"] = last_hidden.new_tensor(weight, dtype=torch.float32)
            losses[f"{channel}_alignment_loss"] = channel_loss
            losses[f"{channel}_weighted_alignment_loss"] = weighted_loss
            total_alignment_loss = total_alignment_loss + weighted_loss
            channel_losses[channel] = channel_loss

        if (
            self.gradnorm_enabled
            and run_gradient_probe
            and training_step is not None
            and int(training_step) > int(self.gradnorm_config["warmup_steps"])
            and (int(training_step) - int(self.gradnorm_config["warmup_steps"]))
            % int(self.gradnorm_config["gradient_probe_interval"])
            == 0
        ):
            losses.update(self._compute_gradnorm_probe(last_hidden, channel_losses))

        losses["alignment_loss"] = total_alignment_loss
        losses["total_loss"] = total_alignment_loss
        return losses
