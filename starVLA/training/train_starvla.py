# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

def debug_on():
    import os
    import sys

    explicit_debug = os.environ.get("STARVLA_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    no_cli_args = len(sys.argv) == 1
    enabled = explicit_debug or no_cli_args
    if not enabled or not no_cli_args:
        return

    # 直接通过 IDE/Python 启动且没有命令行参数时，默认执行第三阶段单步调试：
    # 同时启用 spatial/future/plan 教师，并把数据限制到每个子数据集的前 3 个 episode。
    # 若单卡 DeepSpeed 无法满足环境要求，可由 accelerate 启动并设置
    # STARVLA_DEBUG=1；此时保留 accelerate 已写入的分布式环境变量。
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    debug_env = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in debug_env.items():
        os.environ.setdefault(key, value)

    # 第一阶段（需要时取消注释，并注释下方第三阶段）
    # sys.argv = [
    #     "train_starvla.py",
    #     "--config_yaml",
    #     "./examples/simBenchmarks/LIBERO/train_files/space2act_libero_state.yaml",
    #     "--framework.name", "QwenSpace2ActGR00T",
    #     "--framework.qwenvl.base_vlm", "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
    #     "--framework.latent_reasoning.stage", "1",
    #     "--framework.latent_reasoning.token_init_strategy", "avg",
    #     "--framework.latent_reasoning.channels.spatial.token_enabled", "true",
    #     "--framework.latent_reasoning.channels.spatial.alignment_enabled", "true",
    #     "--framework.latent_reasoning.channels.future.token_enabled", "false",
    #     "--framework.latent_reasoning.channels.future.alignment_enabled", "false",
    #     "--framework.latent_reasoning.channels.plan.token_enabled", "false",
    #     "--framework.latent_reasoning.channels.plan.alignment_enabled", "false",
    #     "--datasets.vla_data.data_root_dir", "/data/jibaixu/Datasets/LIBERO/libero_no_noop",
    #     "--datasets.vla_data.data_mix", "libero_all",
    #     "--datasets.vla_data.episode_indices", "[0,1,2]",
    #     "--datasets.vla_data.teacher_latents.channels.spatial.relative_dir", "lingbot_latent_step4_numlayer24_stride16",
    #     "--datasets.vla_data.per_device_batch_size", "1",
    #     "--datasets.vla_data.video_backend", "torchvision_av",
    #     "--datasets.vla_data.num_workers", "0",
    #     "--trainer.freeze_modules", "",
    #     "--trainer.gradient_accumulation_steps", "1",
    #     "--trainer.max_train_steps", "1",
    #     "--trainer.save_interval", "999999",
    #     "--trainer.eval_interval", "999999",
    #     "--trainer.logging_frequency", "1",
    #     "--run_root_dir", "./playground/Checkpoints",
    #     "--run_id", "debug_space2act_libero_stage1_ep0_2",
    #     "--wandb_project", "starVLA_Libero",
    #     "--wandb_entity", "jinhuiye",
    # ]

    # 第二阶段（需要时取消注释，并注释下方第三阶段）
    # sys.argv = [
    #     "train_starvla.py",
    #     "--config_yaml",
    #     "./examples/simBenchmarks/LIBERO/train_files/space2act_libero_state.yaml",
    #     "--framework.name", "QwenSpace2ActGR00T",
    #     "--framework.qwenvl.base_vlm", "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
    #     "--framework.latent_reasoning.stage", "2",
    #     "--framework.latent_reasoning.token_init_strategy", "avg",
    #     "--framework.latent_reasoning.channels.spatial.token_enabled", "true",
    #     "--framework.latent_reasoning.channels.spatial.alignment_enabled", "true",
    #     "--framework.latent_reasoning.channels.future.token_enabled", "true",
    #     "--framework.latent_reasoning.channels.future.alignment_enabled", "true",
    #     "--framework.latent_reasoning.channels.plan.token_enabled", "false",
    #     "--framework.latent_reasoning.channels.plan.alignment_enabled", "false",
    #     "--datasets.vla_data.data_root_dir", "/data/jibaixu/Datasets/LIBERO/libero_no_noop",
    #     "--datasets.vla_data.data_mix", "libero_all",
    #     "--datasets.vla_data.episode_indices", "[0,1,2]",
    #     "--datasets.vla_data.teacher_latents.channels.spatial.relative_dir", "lingbot_latent_step4_numlayer24_stride16",
    #     "--datasets.vla_data.teacher_latents.channels.future.relative_dir", "wan2_2_future_latent_h16_step20_block20z_stride8_480x480",
    #     "--datasets.vla_data.per_device_batch_size", "1",
    #     "--datasets.vla_data.video_backend", "torchvision_av",
    #     "--datasets.vla_data.num_workers", "0",
    #     # 留空可调试 Stage 2 前向；填写 Stage 1 checkpoint 可同时验证课程迁移。
    #     "--trainer.pretrained_checkpoint", "",
    #     "--trainer.freeze_modules", "",
    #     "--trainer.gradient_accumulation_steps", "1",
    #     "--trainer.max_train_steps", "1",
    #     "--trainer.save_interval", "999999",
    #     "--trainer.eval_interval", "999999",
    #     "--trainer.logging_frequency", "1",
    #     "--run_root_dir", "./playground/Checkpoints",
    #     "--run_id", "debug_space2act_libero_stage2_ep0_2",
    #     "--wandb_project", "starVLA_Libero",
    #     "--wandb_entity", "jinhuiye",
    # ]

    # 第三阶段
    sys.argv = [
        "train_starvla.py",
        "--config_yaml",
        "./examples/simBenchmarks/LIBERO/train_files/space2act_libero_state.yaml",
        "--framework.name", "QwenSpace2ActGR00T",
        "--framework.qwenvl.base_vlm", "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        "--framework.latent_reasoning.stage", "3",
        "--framework.latent_reasoning.token_init_strategy", "avg",
        "--framework.latent_reasoning.channels.spatial.token_enabled", "true",
        "--framework.latent_reasoning.channels.spatial.alignment_enabled", "true",
        "--framework.latent_reasoning.channels.future.token_enabled", "true",
        "--framework.latent_reasoning.channels.future.alignment_enabled", "true",
        "--framework.latent_reasoning.channels.plan.token_enabled", "true",
        "--framework.latent_reasoning.channels.plan.alignment_enabled", "true",
        "--datasets.vla_data.data_root_dir", "/data/jibaixu/Datasets/LIBERO/libero_no_noop",
        "--datasets.vla_data.data_mix", "libero_all",
        "--datasets.vla_data.episode_indices", "[0,1,2]",
        "--datasets.vla_data.teacher_latents.channels.spatial.relative_dir", "lingbot_latent_step4_numlayer24_stride16",
        "--datasets.vla_data.teacher_latents.channels.future.relative_dir", "wan2_2_future_latent_h16_step20_block20z_stride8_480x480",
        "--datasets.vla_data.teacher_latents.channels.plan.relative_dir", "moto_plan_latent_pre_vq_delta4x4_h16_stride1_epoch10",
        "--datasets.vla_data.per_device_batch_size", "1",
        "--datasets.vla_data.video_backend", "torchvision_av",
        "--datasets.vla_data.num_workers", "0",
        # 留空可调试 Stage 3 前向；填写 Stage 2 checkpoint 可同时验证课程迁移。
        "--trainer.pretrained_checkpoint", "",
        "--trainer.freeze_modules", "",
        "--trainer.gradient_accumulation_steps", "1",
        "--trainer.max_train_steps", "1",
        "--trainer.save_interval", "999999",
        "--trainer.eval_interval", "999999",
        "--trainer.logging_frequency", "1",
        "--run_root_dir", "./playground/Checkpoints",
        "--run_id", "debug_space2act_libero_stage3_ep0_2",
        "--wandb_project", "starVLA_Libero",
        "--wandb_entity", "jinhuiye",
    ]

    if "debugpy" not in sys.modules:
        sys.argv.extend(["--is_debug", "True"])
# debug_on()

# Standard Library
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat, load_config_with_base
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.vla_batches_consumed = 0
        self.vla_epoch_count = 0
        self.vla_batches_in_epoch = 0
        self._pending_model_resume: str | None = None
        self._data_iterators_initialized = False
        self._adaptive_loss_accumulator = {}
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader, self.lr_scheduler = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        )
        if self._pending_model_resume:
            self._load_fast_resume_weights(self._pending_model_resume)
            self._load_adaptive_state(self._pending_model_resume)
        self._adjust_lr_scheduler_for_resume()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases (best-effort; must not block training)."""
        self._wandb_enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Resolve a lightweight model-weight resume before distributed setup."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.resume_progress_path = os.path.join(self.checkpoint_dir, "latest_resume.json")
        self._cleanup_stale_training_states()

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_checkpoint, resume_step = self._get_latest_model_checkpoint()
            if resume_checkpoint:
                self.resume_from_checkpoint = resume_checkpoint
                self._pending_model_resume = resume_checkpoint
                self.completed_steps = resume_step
                self._load_light_resume_progress()
                logger.info(
                    "Fast resume loaded model weights from %s at step %s; optimizer state will be reinitialized.",
                    resume_checkpoint,
                    self.completed_steps,
                )
                return

            logger.warning(
                "Resume was requested but no portable model checkpoint was found at %s. Starting a new training run.",
                self.checkpoint_dir,
            )
            self.completed_steps = 0
            # A requested light resume must not silently load an unrelated
            # pretrained checkpoint when no portable checkpoint is available.
            self.resume_from_checkpoint = None
            return

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _cleanup_stale_training_states(self) -> None:
        """Remove obsolete full-state checkpoint trees from the old resume implementation."""
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return
        state_root = Path(self.checkpoint_dir)
        for temporary_path in state_root.glob("steps_*_model.*.tmp"):
            try:
                temporary_path.unlink()
            except OSError as exc:
                logger.warning("Could not remove stale checkpoint temporary file %s: %s", temporary_path, exc)
        for stale_dir in list(state_root.glob(".training_state_step_*.incomplete")) + list(
            state_root.glob("training_state_step_*")
        ):
            if stale_dir.is_dir():
                logger.warning("Removing obsolete full training state: %s", stale_dir)
                shutil.rmtree(stale_dir)
        self.accelerator.wait_for_everyone()

    def _get_latest_model_checkpoint(self) -> tuple[str | None, int]:
        candidates = []
        checkpoint_root = Path(self.checkpoint_dir)
        for path in list(checkpoint_root.glob("steps_*_pytorch_model.pt")) + list(
            checkpoint_root.glob("steps_*_model.safetensors")
        ):
            try:
                step = int(path.name.split("_", 2)[1])
            except (IndexError, ValueError):
                continue
            if path.is_file():
                candidates.append((step, path))
        if not candidates:
            return None, 0
        step, path = max(candidates, key=lambda item: item[0])
        return str(path), step

    def _load_fast_resume_weights(self, checkpoint_path: str) -> None:
        """Load a full model export into a ZeRO-3 model without gathering it in RAM/GPU."""
        rank = self.accelerator.process_index
        checkpoint = {}
        if rank == 0:
            if checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                checkpoint = load_file(checkpoint_path, device="cpu")
            else:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
                except (TypeError, RuntimeError, ValueError):
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_names = [*checkpoint] if rank == 0 else None
        if dist.is_available() and dist.is_initialized():
            names = [checkpoint_names]
            dist.broadcast_object_list(names, src=0)
            checkpoint_names = names[0]
        checkpoint_names = set(checkpoint_names or ())

        model = self.accelerator.unwrap_model(self.model)
        try:
            import deepspeed

            for name, parameter in model.named_parameters():
                if name not in checkpoint_names:
                    continue
                with deepspeed.zero.GatheredParameters([parameter], modifier_rank=0):
                    if rank == 0:
                        parameter.data.copy_(checkpoint[name].to(device=parameter.device, dtype=parameter.dtype))
            for name, buffer in model.named_buffers():
                has_buffer = torch.tensor(
                    int(rank == 0 and name in checkpoint and buffer.shape == checkpoint[name].shape),
                    device=buffer.device,
                    dtype=torch.int32,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(has_buffer, src=0)
                if has_buffer.item():
                    if rank == 0:
                        buffer.data.copy_(checkpoint[name].to(device=buffer.device, dtype=buffer.dtype))
                    if dist.is_available() and dist.is_initialized():
                        dist.broadcast(buffer, src=0)
        finally:
            del checkpoint
        self.accelerator.wait_for_everyone()
        logger.info("Fast resume model weights loaded from %s", checkpoint_path)

    def _adaptive_state_path(self, checkpoint_path: str) -> str:
        """Return the optional adaptive-weight sidecar next to a model export."""
        for suffix in ("_pytorch_model.pt", "_model.safetensors"):
            if checkpoint_path.endswith(suffix):
                return checkpoint_path[: -len(suffix)] + "_adaptive.json"
        return os.path.splitext(checkpoint_path)[0] + "_adaptive.json"

    def _load_adaptive_state(self, checkpoint_path: str | None) -> None:
        """Restore non-parameter adaptive weighting state when present."""
        if not checkpoint_path:
            return
        adaptive_model = self.accelerator.unwrap_model(self.model)
        if not hasattr(adaptive_model, "set_adaptive_state"):
            return
        state_path = self._adaptive_state_path(checkpoint_path)
        if not os.path.isfile(state_path):
            logger.info("No adaptive weighting sidecar found at %s; starting its state fresh", state_path)
            return
        try:
            with open(state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            adaptive_model.set_adaptive_state(state)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to restore adaptive weighting state from {state_path}: {exc}") from exc
        logger.info("Restored adaptive weighting state from %s", state_path)

    def _load_light_resume_progress(self) -> None:
        # Light resume never replays a saved data cursor, including when the
        # sidecar is stale, missing, or malformed.
        self.vla_batches_consumed = 0
        self.vla_epoch_count = 0
        self.vla_batches_in_epoch = 0
        if not os.path.isfile(self.resume_progress_path):
            logger.info("No light resume progress file found; resuming from checkpoint step only")
            return
        try:
            with open(self.resume_progress_path, "r", encoding="utf-8") as handle:
                progress = json.load(handle)
            if not isinstance(progress, dict):
                raise ValueError("light resume progress must be a JSON object")
            if int(progress.get("completed_steps", self.completed_steps)) != self.completed_steps:
                logger.warning("Ignoring stale light resume progress file")
                return
            # Replaying the saved cursor can take hours and provides no exact
            # continuation because optimizer/RNG state is not restored either.
            logger.info("Light resume will start the data iterator from a new epoch")
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid light resume progress: %s", exc)

    def _adjust_lr_scheduler_for_resume(self) -> None:
        if self.completed_steps <= 0:
            return
        logger.info("Advancing fresh LR scheduler to resume step %s", self.completed_steps)
        for _ in range(self.completed_steps):
            self.lr_scheduler.step()

    def _save_checkpoint(self):
        """Save the portable model and a small progress sidecar."""
        try:
            state_dict = self.accelerator.get_state_dict(self.model)
            if self.accelerator.is_main_process:
                save_format = getattr(self.config.trainer, "save_format", "pt")
                checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
                if save_format == "safetensors":
                    from safetensors.torch import save_file
                    model_path = checkpoint_path + "_model.safetensors"
                    temporary_path = model_path + ".tmp"
                    save_file(state_dict, temporary_path)
                    os.replace(temporary_path, model_path)
                elif save_format == "pt":
                    model_path = checkpoint_path + "_pytorch_model.pt"
                    temporary_path = model_path + ".tmp"
                    torch.save(state_dict, temporary_path)
                    os.replace(temporary_path, model_path)
                else:
                    raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

                adaptive_model = self.accelerator.unwrap_model(self.model)
                if hasattr(adaptive_model, "get_adaptive_state"):
                    with open(checkpoint_path + "_adaptive.json", "w", encoding="utf-8") as handle:
                        json.dump(adaptive_model.get_adaptive_state(), handle, indent=2, sort_keys=True)
                with open(self.resume_progress_path + ".tmp", "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "format_version": 3,
                            "completed_steps": self.completed_steps,
                            "data_cursor_policy": "reset_epoch",
                        },
                        handle,
                        indent=2,
                        sort_keys=True,
                    )
                os.replace(self.resume_progress_path + ".tmp", self.resume_progress_path)
                # self._remove_older_model_checkpoints(checkpoint_path) # 删除旧的权重，暂时不删除，磁盘空间充裕
                with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                    f.write(json.dumps({"steps": self.completed_steps}) + "\n")
                self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
                if isinstance(self.config, AccessTrackedConfig):
                    self.config.save_accessed_config(Path(self.config.output_dir) / "config.yaml", use_original_values=False)
        except Exception as exc:
            logger.warning("Checkpoint export at step %s failed; training will continue: %s", self.completed_steps, exc)
        finally:
            self.accelerator.wait_for_everyone()

    def _remove_older_model_checkpoints(self, current_checkpoint_path: str) -> None:
        """Keep only the newest portable model export to limit disk usage."""
        current_prefix = os.path.basename(current_checkpoint_path) + "_"
        checkpoint_root = Path(self.checkpoint_dir)
        for path in checkpoint_root.glob("steps_*"):
            if not path.is_file() or path.name.startswith(current_prefix):
                continue
            if not (
                path.name.endswith("_pytorch_model.pt")
                or path.name.endswith("_model.safetensors")
                or path.name.endswith("_adaptive.json")
            ):
                continue
            try:
                path.unlink()
                logger.info("Removed older model checkpoint: %s", path)
            except OSError as exc:
                logger.warning("Could not remove older model checkpoint %s: %s", path, exc)

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create a fresh data iterator without replaying a saved cursor."""
        self._set_vla_epoch(self.vla_epoch_count)
        self.vla_iter = iter(self.vla_train_dataloader)
        batches_in_epoch = len(self.vla_train_dataloader)
        if batches_in_epoch <= 0:
            raise RuntimeError("Cannot create an iterator from an empty VLA dataloader")
        self._data_iterators_initialized = True

    def _set_vla_epoch(self, epoch: int) -> None:
        """Match the sampler epoch used when the current iterator was first created."""
        set_epoch = getattr(self.vla_train_dataloader, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
            return
        sampler = getattr(self.vla_train_dataloader, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_epoch_count += 1
            self.vla_batches_in_epoch = 0
            self._set_vla_epoch(self.vla_epoch_count)
            self.vla_iter = iter(self.vla_train_dataloader)
            batch_vla = next(self.vla_iter)

        self.vla_batches_consumed += 1
        self.vla_batches_in_epoch += 1
        return batch_vla

    def _perf_rank_timing(self, data_time: float, model_time: float) -> dict[str, float]:
        """Optionally aggregate per-rank timings for short performance probes."""
        if os.environ.get("STARVLA_PERF_RANK_TIMING", "0").lower() not in {"1", "true", "yes", "on"}:
            return {}
        values = torch.tensor([data_time, model_time], device=self.accelerator.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            total = values.clone()
            minimum = values.clone()
            maximum = values.clone()
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            total /= dist.get_world_size()
        else:
            total = minimum = maximum = values
        return {
            "perf/rank_data_mean": float(total[0].item()),
            "perf/rank_data_min": float(minimum[0].item()),
            "perf/rank_data_max": float(maximum[0].item()),
            "perf/rank_model_mean": float(total[1].item()),
            "perf/rank_model_min": float(minimum[1].item()),
            "perf/rank_model_max": float(maximum[1].item()),
        }

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        if not self._data_iterators_initialized:
            self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            perf_interval = int(os.environ.get("STARVLA_PERF_RANK_TIMING_INTERVAL", "10"))
            if self.completed_steps > 0 and perf_interval > 0 and self.completed_steps % perf_interval == 0:
                step_metrics.update(
                    self._perf_rank_timing(
                        t_end_data - t_start_data,
                        t_end_model - t_start_model,
                    )
                )
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # All micro-batches in this accumulation window use the same logical
            # optimizer step; completed_steps is incremented after this method.
            optimizer_step = self.completed_steps + 1

            with torch.autocast("cuda", dtype=torch.bfloat16):
                forward_kwargs = {}
                if str(self.config.framework.name) == "QwenSpace2ActGR00T":
                    # Progressive Space2Act 以 optimization step（而非 micro-batch）
                    # 插值教师权重；legacy 配置会收到该值但继续使用固定 loss_weight。
                    # The final micro-batch of an accumulation window belongs to
                    # the step that will be logged after optimizer.step().
                    #! 损失调度fixed模式：训练外层向模型传入训练步，实际损失权重调度过程发生在模型内部 【本质是线性插值】
                    forward_kwargs["training_step"] = optimizer_step
                    # 梯度累积期间只在同步 micro-batch 做一次诊断，避免
                    # 同一 optimization step 重复执行 autograd.grad。
                    forward_kwargs["run_gradient_diagnostics"] = self.accelerator.sync_gradients
                    # GradNorm probe 与诊断共享同步边界；在当前 ZeRO-2 配置中
                    # gradient_accumulation_steps 固定为 1，因此每个同步 batch
                    # 就是一个 optimizer step。
                    forward_kwargs["run_gradient_probe"] = self.accelerator.sync_gradients
                output_dict = self.model.forward(batch_vla, **forward_kwargs)
                action_loss = output_dict["action_loss"]
                # Space2Act 的 total_loss = 动作损失 + 各教师对齐损失；基础框架
                # 没有该字段时回退到 action_loss，使训练循环保持向后兼容。
                total_loss = output_dict.get("total_loss", action_loss)

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

            # DWA/GradNorm 的非参数状态在 optimizer step 完成后统一推进。
            adaptive_model = self.accelerator.unwrap_model(self.model)
            if getattr(adaptive_model, "dwa_enabled", False):
                for channel in adaptive_model.dwa_channels:
                    key = f"{channel}_alignment_loss"
                    if key in output_dict:
                        self._adaptive_loss_accumulator.setdefault(channel, []).append(
                            output_dict[key].detach()
                        )

            if self.accelerator.sync_gradients and hasattr(adaptive_model, "update_adaptive_state"):
                channel_losses = {
                    channel: torch.stack(values).mean()
                    for channel, values in self._adaptive_loss_accumulator.items()
                    if values
                }
                channel_gradient_norms = {
                    channel: output_dict[f"gradnorm_probe/{channel}"].detach()
                    for channel in getattr(adaptive_model, "gradnorm_channels", ())
                    if f"gradnorm_probe/{channel}" in output_dict
                }
                adaptive_metrics = adaptive_model.update_adaptive_state(
                    optimizer_step,
                    channel_losses,
                    should_update=True,
                    channel_gradient_norms=channel_gradient_norms,
                )
                self._adaptive_loss_accumulator.clear()
            else:
                adaptive_metrics = {}

        metrics = {"action_dit_loss": action_loss.item(), "total_loss": total_loss.item()}
        # 自动收集框架返回的标量（如各通道 alignment loss），避免训练器
        # 为后续 future/plan 通道逐项增加硬编码日志字段。
        for name, value in output_dict.items():
            if name in {"action_loss", "total_loss"}:
                continue
            if torch.is_tensor(value) and value.numel() == 1:
                metrics[name] = value.detach().item()
        metrics.update(adaptive_metrics)
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        # Final checkpoint gathering has the same all-rank requirement.
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    # 在模型构造前设置一次 seed，确保新增 token/alignment head 的初始化可复现；
    # prepare_training 中会再次重置，从而让训练期 RNG 不受初始化耗用量影响。
    rank = dist.get_rank() if dist.is_initialized() else 0
    seed = cfg.seed + rank if hasattr(cfg, "seed") else rank + 3047
    set_seed(seed)

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = load_config_with_base(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
