"""在 Qwen3-VL + GR00T 上加入紧凑的 S/F/P 类型化 latent reasoning token。"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig, Qwen_GR00T
from starVLA.model.modules.latent_reasoning import CrossAttentionLatentAligner
from starVLA.model.tools import FRAMEWORK_REGISTRY


_SPACE2ACT_CHANNELS = ("spatial", "future", "plan")


def _is_enabled(value) -> bool:
    return value in (True, "True", "true", 1)


def _normalize_loss_schedule(schedule_cfg, active_channels) -> tuple[tuple[int, dict[str, float]], ...]:
    """校验 progressive knots，并转换为只含 Python 标量的不可变结构。"""
    if not schedule_cfg or not _is_enabled(schedule_cfg.get("enabled", False)):
        return ()

    expected_channels = set(_SPACE2ACT_CHANNELS)
    active_channels = set(active_channels)
    if active_channels != expected_channels:
        raise ValueError(
            "Progressive Space2Act loss_schedule requires all spatial/future/plan "
            f"alignment channels, got {sorted(active_channels)}"
        )

    raw_knots = schedule_cfg.get("knots", None)
    if not raw_knots:
        raise ValueError("Progressive Space2Act loss_schedule requires non-empty knots")

    normalized = []
    previous_step = None
    for index, knot in enumerate(raw_knots):
        raw_step = knot.get("step", None)
        if raw_step is None or isinstance(raw_step, bool) or int(raw_step) != float(raw_step):
            raise ValueError(f"loss_schedule knot {index} has invalid step: {raw_step!r}")
        step = int(raw_step)
        if step < 0 or (previous_step is not None and step <= previous_step):
            raise ValueError("loss_schedule knot steps must be non-negative and strictly increasing")

        raw_weights = knot.get("weights", None)
        if raw_weights is None or set(raw_weights.keys()) != expected_channels:
            actual = sorted(raw_weights.keys()) if raw_weights is not None else []
            raise ValueError(
                "Every loss_schedule knot must define exactly spatial/future/plan weights, "
                f"got {actual}"
            )
        weights = {str(channel): float(raw_weights[channel]) for channel in _SPACE2ACT_CHANNELS}
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
            raise ValueError("loss_schedule weights must be finite and non-negative")
        normalized.append((step, weights))
        previous_step = step

    if normalized[0][0] != 0:
        raise ValueError("loss_schedule must start at step 0")
    return tuple(normalized)


def _interpolate_loss_schedule(
    knots: tuple[tuple[int, dict[str, float]], ...], training_step: int
) -> dict[str, float]:
    """在相邻 knots 间按 optimization step 线性插值，范围外保持端点值。"""
    if isinstance(training_step, bool) or int(training_step) != float(training_step):
        raise ValueError(f"training_step must be an integer, got {training_step!r}")
    step = max(0, int(training_step))
    steps = [knot_step for knot_step, _ in knots]
    right = bisect_right(steps, step)
    if right == 0:
        return dict(knots[0][1])
    if right >= len(knots):
        return dict(knots[-1][1])

    left_step, left_weights = knots[right - 1]
    right_step, right_weights = knots[right]
    ratio = (step - left_step) / (right_step - left_step)
    return {
        channel: left_weights[channel]
        + ratio * (right_weights[channel] - left_weights[channel])
        for channel in _SPACE2ACT_CHANNELS
    }


def _channel_defaults(token_enabled: bool, alignment_enabled: bool, target_tokens, target_dim):
    """生成三类教师通道共享的默认配置，阶段差异只由开关和目标 shape 表达。"""
    return {
        "token_enabled": token_enabled,
        "token_count": 32,
        "alignment_enabled": alignment_enabled,
        "target_tokens": target_tokens,
        "target_dim": target_dim,
        "alignment_dim": 1024,
        "num_heads": 8,
        "loss_weight": 1.0,
        "mse_weight": 0.1,
    }


@dataclass
class QwenSpace2ActGR00TDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenSpace2ActGR00T"
    latent_reasoning: dict = field(
        default_factory=lambda: {
            "stage": 1,
            "routing_mode": "native_causal",
            "token_init_strategy": "avg",
            "query_chunk_size": 128,
            "loss_schedule": {
                "enabled": False,
                "diagnostic_interval": 0,
                "knots": [],
            },
            "channels": {
                "spatial": _channel_defaults(True, True, 1521, 5120),
                "future": _channel_defaults(False, False, 1125, 3072),
                # 四段 delta-4 Moto pre-VQ latent 按时间拼接为 [32, 32]。
                "plan": _channel_defaults(False, False, 32, 32),
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenSpace2ActGR00T")
class QwenSpace2Act_GR00T(Qwen_GR00T):
    """训练期增加三教师 latent 对齐，同时保持基础 GR00T action head 接口不变。"""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = merge_framework_config(QwenSpace2ActGR00TDefaultConfig, config)
        super().__init__(config=config, **kwargs)
        latent_cfg = self.config.framework.latent_reasoning
        self.space2act_stage = int(latent_cfg.get("stage", 1))
        if self.space2act_stage not in (1, 2, 3):
            raise ValueError(f"Space2Act stage must be 1, 2, or 3, got {self.space2act_stage}")
        # 首版只支持 Qwen 原生因果路由。若配置写入尚未实现的 mask 模式，
        # 在模型构造阶段直接失败，避免训练时产生与预期不同的信息流。
        if latent_cfg.routing_mode != "native_causal":
            raise ValueError("The first Space2Act implementation only supports routing_mode=native_causal")

        # 先扩展 tokenizer/model embedding，再根据启用通道创建训练期 alignment head。
        self.qwen_vl_interface.configure_space2act_tokens(
            latent_cfg.channels, init_strategy=str(latent_cfg.token_init_strategy)
        )
        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        self.alignment_heads = nn.ModuleDict()
        self.alignment_loss_weights: dict[str, float] = {}
        for channel, channel_cfg in latent_cfg.channels.items():
            if channel_cfg.get("alignment_enabled", False) not in (True, "True", "true", 1):
                continue
            if channel_cfg.get("token_enabled", False) not in (True, "True", "true", 1):
                raise ValueError(
                    f"Space2Act channel '{channel}' cannot enable alignment while its tokens are disabled"
                )
            target_tokens = channel_cfg.get("target_tokens", None)
            target_dim = channel_cfg.get("target_dim", None)
            if target_tokens in (None, "null") or target_dim in (None, "null"):
                raise ValueError(
                    f"Space2Act channel '{channel}' enables alignment before target_tokens/target_dim are defined"
                )
            # 每个通道拥有独立的可学习 query 和输出投影，因此空间、未来、规划教师
            # 可以使用不同 token 数和特征维度，互不共享映射参数。
            self.alignment_heads[str(channel)] = CrossAttentionLatentAligner(
                input_dim=hidden_dim,
                target_tokens=int(target_tokens),
                target_dim=int(target_dim),
                alignment_dim=int(channel_cfg.get("alignment_dim", 1024)),
                num_heads=int(channel_cfg.get("num_heads", 8)),
                query_chunk_size=int(latent_cfg.query_chunk_size),
                mse_weight=float(channel_cfg.get("mse_weight", 0.1)),
            )
            self.alignment_loss_weights[str(channel)] = float(channel_cfg.get("loss_weight", 1.0))

        schedule_cfg = latent_cfg.get("loss_schedule", {})
        self.loss_schedule_knots = _normalize_loss_schedule(
            schedule_cfg, self.alignment_heads.keys()
        )
        self.loss_schedule_enabled = bool(self.loss_schedule_knots)
        self.loss_diagnostic_interval = int(schedule_cfg.get("diagnostic_interval", 0))
        if self.loss_diagnostic_interval < 0:
            raise ValueError("loss_schedule diagnostic_interval must be non-negative")

    def _typed_positions(self, input_ids: torch.Tensor, channel: str) -> torch.Tensor:
        """返回每个样本中指定 typed token 的位置，形状为 [B,K_channel]。"""
        token_ids = self.qwen_vl_interface.space2act_token_ids[channel]
        rows = []
        for batch_index in range(input_ids.shape[0]):
            positions = []
            for token_id in token_ids:
                # 每个类型化 token 在单条 prompt 中应恰好出现一次；重复或缺失都说明
                # prompt 构造与 token bank 配置已经失配，不能继续做位置对齐。
                matches = torch.nonzero(input_ids[batch_index] == token_id, as_tuple=False).flatten()
                if matches.numel() != 1:
                    raise ValueError(
                        f"Expected exactly one '{channel}' token id {token_id}, found {matches.numel()}"
                    )
                positions.append(matches[0])
            rows.append(torch.stack(positions))
        return torch.stack(rows)

    def _typed_hidden(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        channel: str,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """按 token id 从 Qwen 最后一层提取指定通道，返回 [B,K_channel,H_vlm]。"""
        if positions is None:
            positions = self._typed_positions(input_ids, channel)
        gather_index = positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return torch.gather(hidden, dim=1, index=gather_index)

    def _effective_alignment_weights(self, training_step: int | None) -> dict[str, float]:
        if not self.loss_schedule_enabled:
            return dict(self.alignment_loss_weights)
        if training_step is None:
            raise ValueError("training_step is required when Space2Act loss_schedule is enabled")
        return _interpolate_loss_schedule(self.loss_schedule_knots, training_step)

    def _compute_gradient_diagnostics(
        self,
        last_hidden: torch.Tensor,
        action_loss: torch.Tensor,
        channel_losses: dict[str, torch.Tensor],
        channel_positions: dict[str, torch.Tensor],
        effective_weights: dict[str, float],
    ) -> dict[str, torch.Tensor]:
        """度量各辅助目标进入 Qwen 最后一层的梯度尺度与 action 冲突。"""
        action_grad = torch.autograd.grad(
            action_loss, last_hidden, retain_graph=True, create_graph=False
        )[0].detach().float()
        diagnostics = {
            "gradient_norm/action_hidden": torch.linalg.vector_norm(action_grad),
        }
        aggregate_aux_grad = torch.zeros_like(action_grad)
        eps = torch.finfo(torch.float32).eps

        for channel, channel_loss in channel_losses.items():
            weighted_loss = effective_weights[channel] * channel_loss
            auxiliary_grad = torch.autograd.grad(
                weighted_loss, last_hidden, retain_graph=True, create_graph=False
            )[0].detach().float()
            aggregate_aux_grad.add_(auxiliary_grad)

            positions = channel_positions[channel]
            gather_index = positions.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
            action_typed_grad = torch.gather(action_grad, 1, gather_index).reshape(-1)
            auxiliary_typed_grad = torch.gather(auxiliary_grad, 1, gather_index).reshape(-1)
            action_norm = torch.linalg.vector_norm(action_typed_grad)
            auxiliary_norm = torch.linalg.vector_norm(auxiliary_typed_grad)
            cosine = torch.dot(action_typed_grad, auxiliary_typed_grad) / (
                action_norm * auxiliary_norm + eps
            )
            diagnostics[f"gradient_norm/{channel}_alignment_hidden"] = auxiliary_norm
            diagnostics[f"gradient_cosine/{channel}_vs_action"] = cosine

        aggregate_norm = torch.linalg.vector_norm(aggregate_aux_grad)
        action_norm = diagnostics["gradient_norm/action_hidden"]
        diagnostics["gradient_norm/alignment_hidden"] = aggregate_norm
        diagnostics["gradient_ratio/alignment_to_action_hidden"] = aggregate_norm / (
            action_norm + eps
        )
        return {name: value.detach() for name, value in diagnostics.items()}

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        training_step = kwargs.pop("training_step", None)
        run_gradient_diagnostics = kwargs.pop("run_gradient_diagnostics", True)
        effective_weights = self._effective_alignment_weights(training_step)
        # 先保持基础 QwenGR00T 的样本字段约定，再额外读取 dataloader 注入的 teacher。
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # Qwen 一次前向同时产生 action head 条件和 S/F/P 对齐所需的 hidden state，
        # 避免为辅助监督重复运行视觉语言 backbone。
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(batch_images, instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        # 逐通道计算教师对齐损失。teacher tensor 保持在 CPU，aligner 内部按 query
        # chunk 搬到 GPU，从而限制 1521x5120 等大目标的瞬时显存占用。
        losses = {}
        total_alignment_loss = last_hidden.new_zeros((), dtype=torch.float32)
        channel_losses = {}
        channel_positions = {}
        for channel, aligner in self.alignment_heads.items():
            weight = effective_weights[channel]
            losses[f"{channel}_alignment_weight"] = last_hidden.new_tensor(
                weight, dtype=torch.float32
            )
            active = not self.loss_schedule_enabled or weight > 0.0
            losses[f"{channel}_alignment_active"] = last_hidden.new_tensor(
                float(active), dtype=torch.float32
            )
            if not active:
                losses[f"{channel}_alignment_loss"] = last_hidden.new_zeros((), dtype=torch.float32)
                losses[f"{channel}_weighted_alignment_loss"] = last_hidden.new_zeros(
                    (), dtype=torch.float32
                )
                continue

            teacher_key = "space2act_teacher"
            if any(teacher_key not in example or channel not in example[teacher_key] for example in examples):
                raise KeyError(f"Missing Space2Act teacher tensors for enabled channel '{channel}'")
            positions = self._typed_positions(qwen_inputs["input_ids"], channel)
            compact_hidden = self._typed_hidden(
                qwen_inputs["input_ids"], last_hidden, channel, positions=positions
            )
            teachers = [example[teacher_key][channel] for example in examples]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                channel_loss = aligner.compute_loss(compact_hidden, teachers)
            weighted_channel_loss = weight * channel_loss
            losses[f"{channel}_alignment_loss"] = channel_loss
            losses[f"{channel}_weighted_alignment_loss"] = weighted_channel_loss
            total_alignment_loss = total_alignment_loss + weighted_channel_loss
            channel_losses[channel] = channel_loss
            channel_positions[channel] = positions

        # action 分支继续沿用 GR00T 的 repeated diffusion 训练方式；attention mask、
        # state 和 Qwen hidden 按相同 repeats 扩展，保持 batch 维严格一致。
        with torch.autocast("cuda", dtype=torch.float32):
            action_tensor = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            action_target = action_tensor[:, -self.action_horizon :, :]
            repeats = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            action_target = action_target.repeat(repeats, 1, 1)
            action_hidden = last_hidden.repeat(repeats, 1, 1)
            action_mask = None
            if backbone_attention_mask is not None:
                action_mask = backbone_attention_mask.repeat(repeats, 1).to(dtype=torch.bool)
            state_repeated = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state_tensor.repeat(repeats, 1, 1)
            action_loss = self.action_model(
                action_hidden, action_target, state_repeated, encoder_attention_mask=action_mask
            )

        if self.loss_schedule_enabled:
            steps = [step for step, _ in self.loss_schedule_knots]
            segment = max(0, min(bisect_right(steps, int(training_step)) - 1, len(steps) - 1))
            losses["loss_schedule_segment"] = last_hidden.new_tensor(
                float(segment), dtype=torch.float32
            )
            should_diagnose = (
                bool(run_gradient_diagnostics)
                and self.loss_diagnostic_interval > 0
                and int(training_step) > 0
                and int(training_step) % self.loss_diagnostic_interval == 0
                and bool(channel_losses)
                and torch.is_grad_enabled()
                and last_hidden.requires_grad
            )
            if should_diagnose:
                losses.update(
                    self._compute_gradient_diagnostics(
                        last_hidden,
                        action_loss,
                        channel_losses,
                        channel_positions,
                        effective_weights,
                    )
                )

        # trainer 优先反传 total_loss；同时保留分项，便于独立观察 action 与各教师损失。
        losses["action_loss"] = action_loss
        losses["alignment_loss"] = total_alignment_loss
        losses["total_loss"] = action_loss + total_alignment_loss
        return losses

    def adapt_checkpoint_state_dict(self, checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """适配 base/前一阶段 checkpoint，并严格约束课程迁移与同阶段恢复。"""
        current = self.state_dict()
        adapted = dict(checkpoint)
        allowed_suffixes = ("embed_tokens.weight", "lm_head.weight")
        # 带 alignment_heads 的 checkpoint 视为 Space2Act 续训；否则视为基础模型迁移。
        # 基础 checkpoint 没有 S/F/P token，其同 shape 词表也不能覆盖当前初始化值。
        is_space2act_checkpoint = any(key.startswith("alignment_heads.") for key in checkpoint)
        protected_token_ids = sorted(
            token_id
            for token_ids in self.qwen_vl_interface.space2act_token_ids.values()
            for token_id in token_ids
        )
        for key, source in checkpoint.items():
            target = current.get(key)
            if target is None:
                continue
            is_vocab_matrix = key.endswith(allowed_suffixes)
            can_copy_vocab = (
                is_vocab_matrix
                and source.ndim == target.ndim
                and source.shape[1:] == target.shape[1:]
                and source.shape[0] <= target.shape[0]
            )
            if can_copy_vocab and (source.shape != target.shape or not is_space2act_checkpoint):
                # 先以当前扩展后矩阵为底，再复制旧词表行。加载基础 checkpoint 时，
                # 最后恢复受保护 token 行，保留 avg/zero/normal 初始化策略的结果。
                expanded = target.detach().cpu().clone()
                expanded[: source.shape[0]].copy_(source)
                if not is_space2act_checkpoint:
                    valid_ids = [token_id for token_id in protected_token_ids if token_id < target.shape[0]]
                    expanded[valid_ids] = target.detach().cpu()[valid_ids]
                adapted[key] = expanded

        stage = int(getattr(self, "space2act_stage", 1))
        if stage >= 2:
            if not is_space2act_checkpoint:
                raise ValueError(
                    f"Space2Act Stage {stage} must warm-start from a Space2Act checkpoint"
                )

            checkpoint_channels = {
                key.split(".", 2)[1]
                for key in checkpoint
                if key.startswith("alignment_heads.") and len(key.split(".", 2)) == 3
            }
            active_channels = set(self.alignment_heads.keys())
            predecessor_channels, new_channel = {
                2: ({"spatial"}, "future"),
                3: ({"spatial", "future"}, "plan"),
            }[stage]

            if checkpoint_channels == active_channels:
                # 同阶段 checkpoint 必须完整匹配当前 mapper 集合，供正常 resume/restart。
                new_channels = set()
            elif checkpoint_channels == predecessor_channels:
                expected_active = predecessor_channels | {new_channel}
                if active_channels != expected_active:
                    raise ValueError(
                        f"Stage {stage} warm-start requires active alignment channels "
                        f"{sorted(expected_active)}, got {sorted(active_channels)}"
                    )
                new_channels = {new_channel}
            else:
                raise ValueError(
                    f"Stage {stage} expects a predecessor checkpoint with alignment channels "
                    f"{sorted(predecessor_channels)} or a same-stage checkpoint with "
                    f"{sorted(active_channels)}, got {sorted(checkpoint_channels)}"
                )

            for key, value in current.items():
                if any(key.startswith(f"alignment_heads.{channel}.") for channel in new_channels):
                    adapted[key] = value.detach().cpu().clone()

            missing = sorted(set(current) - set(adapted))
            unexpected = sorted(set(adapted) - set(current))
            if missing or unexpected:
                raise ValueError(
                    f"Checkpoint is incompatible with Space2Act Stage {stage}: "
                    f"missing={missing[:10]}, unexpected={unexpected[:10]}"
                )
        return adapted
