# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from typing import Optional

import torch
from starVLA.model.tools import has_flash_attn  # unified flash-attn detection (GPU / NPU)
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN = 151669  # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = (
    153716  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
)


import torch.nn as nn


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        attn_implementation = "sdpa"
        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            if not has_flash_attn():
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, # resize image no longer needed? @TODO check bug
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

        # token_ids 保存 S/F/P 三个类型化 token bank 的词表 id；active_channels
        # 只控制本轮实验实际插入哪些 bank，二者分离便于阶段扩展和 checkpoint 兼容。
        self.space2act_token_ids: dict[str, list[int]] = {}
        self.space2act_active_channels: set[str] = set()
        self.space2act_action_token_ids: list[int] = []
        self._space2act_action_token_text = ""
        self.space2act_token_order = "spatial_instruction_future_plan"

    def configure_space2act_tokens(
        self,
        channel_configs,
        init_strategy: str = "avg",
        action_token_count: int = 0,
        token_order: str = "spatial_instruction_future_plan",
    ) -> None:
        """注册 S/F/P 与可选 A token bank，并记录当前阶段启用的通道。"""
        supported_orders = {
            "spatial_instruction_future_plan",
            "spatial_instruction_plan_future",
        }
        if token_order not in supported_orders:
            raise ValueError(
                "Unsupported Space2Act token_order: "
                f"{token_order}. Expected one of {sorted(supported_orders)}"
            )
        # 即使 Stage 1 只启用 spatial，也一次性注册三类 token。这样后续 Stage 2/3
        # 切换通道时词表大小和 token id 保持稳定，已有 checkpoint 可以直接续训。
        channel_prefixes = {"spatial": "s", "future": "f", "plan": "p"}
        all_tokens: list[str] = []
        tokens_by_channel: dict[str, list[str]] = {}
        for channel, prefix in channel_prefixes.items():
            channel_cfg = channel_configs.get(channel, {})
            token_count = int(channel_cfg.get("token_count", 32))
            tokens = [f"<|space2act_{prefix}_{index:03d}|>" for index in range(token_count)]
            tokens_by_channel[channel] = tokens
            all_tokens.extend(tokens)
        if action_token_count < 0:
            raise ValueError("action_token_count must be non-negative")
        action_tokens = [
            f"<|space2act_a_{index:03d}|>" for index in range(int(action_token_count))
        ]
        all_tokens.extend(action_tokens)

        # tokenizer 与模型 embedding 必须同步扩容。只把词表中真正不存在的 token
        # 记为 new token，避免恢复 Space2Act checkpoint 时重复初始化已有向量。
        tokenizer = self.processor.tokenizer
        previous_vocab = tokenizer.get_vocab()
        tokens_to_initialize = [token for token in all_tokens if token not in previous_vocab]
        old_size = self.model.get_input_embeddings().weight.shape[0]
        tokenizer.add_special_tokens({"additional_special_tokens": all_tokens})
        target_size = max(len(tokenizer), old_size)
        if target_size > old_size:
            self.model.resize_token_embeddings(target_size)

        new_token_ids = [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens_to_initialize]
        if new_token_ids:
            input_embedding = self.model.get_input_embeddings().weight
            output_module = self.model.get_output_embeddings()
            output_embedding = output_module.weight if output_module is not None else None
            reference_size = min(len(previous_vocab), old_size)
            with torch.no_grad():
                # avg 是默认的平稳初始化；zero 和 normal 用于消融实验。若输入/输出
                # embedding 未共享权重，两侧必须使用相同策略分别初始化。
                if init_strategy == "avg":
                    input_value = input_embedding[:reference_size].detach().float().mean(dim=0)
                    input_embedding[new_token_ids] = input_value.to(input_embedding.dtype)
                    if output_embedding is not None and output_embedding.data_ptr() != input_embedding.data_ptr():
                        output_value = output_embedding[:reference_size].detach().float().mean(dim=0)
                        output_embedding[new_token_ids] = output_value.to(output_embedding.dtype)
                elif init_strategy == "zero":
                    input_embedding[new_token_ids] = 0
                    if output_embedding is not None and output_embedding.data_ptr() != input_embedding.data_ptr():
                        output_embedding[new_token_ids] = 0
                elif init_strategy == "normal":
                    input_embedding[new_token_ids] = torch.randn(
                        len(new_token_ids), input_embedding.shape[1],
                        device=input_embedding.device, dtype=input_embedding.dtype,
                    ) * 0.02
                    if output_embedding is not None and output_embedding.data_ptr() != input_embedding.data_ptr():
                        output_embedding[new_token_ids] = torch.randn(
                            len(new_token_ids), output_embedding.shape[1],
                            device=output_embedding.device, dtype=output_embedding.dtype,
                        ) * 0.02
                else:
                    raise ValueError(f"Unsupported Space2Act token_init_strategy: {init_strategy}")

        # 同时保存数值 id（用于从 hidden state 精确取位）和拼接文本（用于构造 prompt）。
        self.space2act_token_ids = {
            channel: [int(tokenizer.convert_tokens_to_ids(token)) for token in tokens]
            for channel, tokens in tokens_by_channel.items()
        }
        self._space2act_token_text = {
            channel: "".join(tokens) for channel, tokens in tokens_by_channel.items()
        }
        self.space2act_action_token_ids = [
            int(tokenizer.convert_tokens_to_ids(token)) for token in action_tokens
        ]
        self._space2act_action_token_text = "".join(action_tokens)
        self.space2act_token_order = token_order
        # token_enabled 只决定 forward 中是否插入，不影响已经注册的完整词表。
        self.space2act_active_channels = {
            channel
            for channel in channel_prefixes
            if channel_configs.get(channel, {}).get("token_enabled", False) in (True, "True", "true", 1)
        }

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            # 使用 Qwen 原生 causal mask，不额外构造 block mask。S 始终位于图像后、
            # 指令前；P/F 的相对顺序由 checkpoint-visible token_order 决定。
            if "spatial" in self.space2act_active_channels:
                content.append({"type": "text", "text": self._space2act_token_text["spatial"]})
            content.append({"type": "text", "text": prompt})
            ordered_suffixes = (
                ("future", "plan")
                if self.space2act_token_order == "spatial_instruction_future_plan"
                else ("plan", "future")
            )
            for channel in ordered_suffixes:
                if channel in self.space2act_active_channels:
                    content.append({"type": "text", "text": self._space2act_token_text[channel]})
            # A tokens 位于全部 reasoning token 之后，用同一次 causal forward 汇聚
            # action intent；它们不属于教师通道，也不建立 alignment target。
            if self.space2act_action_token_ids:
                content.append({"type": "text", "text": self._space2act_action_token_text})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference

        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:  #  here only for fast_tokenizer now.
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            labels[labels == self.processor.tokenizer.pad_token_id] = -100  ## mask out pad tokens as well
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
