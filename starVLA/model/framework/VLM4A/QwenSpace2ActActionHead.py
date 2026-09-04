"""Frozen-VLM Action Head stage for the independent Space2Act pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig, Qwen_GR00T
from starVLA.model.framework.VLM4A.space2act_two_stage import SPACE2ACT_CHANNELS, is_enabled
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _channel_defaults(target_tokens: int, target_dim: int) -> dict:
    return {
        "token_enabled": True,
        "token_count": 32,
        # This framework never creates mapper heads or reads teacher tensors.
        "alignment_enabled": False,
        "target_tokens": target_tokens,
        "target_dim": target_dim,
    }


@dataclass
class QwenSpace2ActActionHeadDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenSpace2ActActionHead"
    latent_reasoning: dict = field(
        default_factory=lambda: {
            "routing_mode": "native_causal",
            "token_order": "spatial_instruction_plan_future",
            "token_init_strategy": "avg",
            "channels": {
                "spatial": _channel_defaults(1521, 5120),
                "future": _channel_defaults(1125, 3072),
                "plan": _channel_defaults(32, 32),
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenSpace2ActActionHead")
class QwenSpace2Act_ActionHead(Qwen_GR00T):
    """Train GR00T on actions while keeping a transferred typed-token VLM frozen.

    No alignment mappers, teacher cache, PCGrad, routing, or adaptive loss state
    is constructed by this class.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = merge_framework_config(QwenSpace2ActActionHeadDefaultConfig, config)
        latent_cfg = config.framework.latent_reasoning
        if str(latent_cfg.get("routing_mode", "native_causal")) != "native_causal":
            raise ValueError("QwenSpace2ActActionHead only supports routing_mode=native_causal")
        invalid = [
            channel
            for channel in SPACE2ACT_CHANNELS
            if not is_enabled(latent_cfg.channels[channel].get("token_enabled", False))
            or is_enabled(latent_cfg.channels[channel].get("alignment_enabled", False))
        ]
        if invalid:
            raise ValueError(
                "QwenSpace2ActActionHead requires all typed tokens and no alignment; invalid="
                f"{invalid}"
            )
        super().__init__(config=config, **kwargs)
        self.qwen_vl_interface.configure_space2act_tokens(
            latent_cfg.channels,
            init_strategy=str(latent_cfg.get("token_init_strategy", "avg")),
            action_token_count=0,
            token_order=str(latent_cfg.get("token_order", "spatial_instruction_future_plan")),
        )

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

    def validate_pretrained_checkpoint(self, checkpoint_path, reload_modules=None) -> None:
        requested = {item.strip() for item in (reload_modules or "").split(",") if item.strip()}
        if "qwen_vl_interface" not in requested:
            return
        checkpoint = Path(checkpoint_path).resolve()
        candidates = (checkpoint.parent / "config.full.yaml", checkpoint.parent.parent / "config.full.yaml")
        source_path = next((path for path in candidates if path.is_file()), None)
        if source_path is None:
            raise ValueError(
                "Space2Act action-head transfer requires config.full.yaml in the VLM run output"
            )
        source_framework = OmegaConf.load(source_path).get("framework", {})
        if source_framework.get("name") != "QwenSpace2ActVLM":
            raise ValueError(
                "qwen_vl_interface transfer requires a QwenSpace2ActVLM distillation checkpoint"
            )
        source_latent = source_framework.get("latent_reasoning", {})
        actual = {
            "base_vlm": str(source_framework.get("qwenvl", {}).get("base_vlm", "")),
            "token_order": str(source_latent.get("token_order", "spatial_instruction_future_plan")),
            "channels": {
                channel: int(source_latent.get("channels", {}).get(channel, {}).get("token_count", -1))
                for channel in SPACE2ACT_CHANNELS
            },
        }
        if actual != self.space2act_layout():
            raise ValueError(
                "Incompatible Space2Act VLM checkpoint layout for action-head training: "
                f"expected {self.space2act_layout()}, got {actual}"
            )
