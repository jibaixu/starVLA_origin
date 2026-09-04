"""Train a VLM-only Space2Act distillation checkpoint on the VLA dataloader."""

import argparse

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.share_tools import apply_config_compat, load_config_with_base
from starVLA.training.train_starvla import (
    VLATrainer,
    accelerator,
    build_framework,
    dist,
    logger,
    prepare_data,
    set_seed,
    setup_directories,
    setup_optimizer_and_scheduler,
    wrap_config,
)
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


class Space2ActVLMTrainer(VLATrainer):
    """Reuse distributed/checkpoint plumbing while replacing action-loss training."""

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """The distillation model intentionally has no action head to evaluate."""
        return step_metrics or {}

    def _train_step(self, batch_vla, batch_vlm=None):
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            optimizer_step = self.completed_steps + 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(
                    batch_vla,
                    training_step=optimizer_step,
                    run_gradient_probe=self.accelerator.sync_gradients,
                )
                total_loss = output_dict["total_loss"]
            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

            adaptive_model = self.accelerator.unwrap_model(self.model)
            if self.accelerator.sync_gradients and hasattr(adaptive_model, "update_adaptive_state"):
                gradient_norms = {
                    channel: output_dict[f"gradnorm_probe/{channel}"].detach()
                    for channel in getattr(adaptive_model, "gradnorm_channels", ())
                    if f"gradnorm_probe/{channel}" in output_dict
                }
                adaptive_metrics = adaptive_model.update_adaptive_state(
                    optimizer_step,
                    channel_gradient_norms=gradient_norms,
                )
            else:
                adaptive_metrics = {}

        metrics = {"alignment_loss": total_loss.detach().item(), "total_loss": total_loss.detach().item()}
        for name, value in output_dict.items():
            if name != "total_loss" and torch.is_tensor(value) and value.numel() == 1:
                metrics[name] = value.detach().item()
        metrics.update(adaptive_metrics)
        return metrics


def main(cfg) -> None:
    logger.info("Space2Act VLM distillation :: Warming Up")
    rank = dist.get_rank() if dist.is_initialized() else 0
    set_seed(cfg.seed + rank if hasattr(cfg, "seed") else rank + 3047)
    cfg = wrap_config(cfg)

    output_dir = setup_directories(cfg=cfg)
    model = build_framework(cfg)
    dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=model, cfg=cfg)
    trainer = Space2ActVLMTrainer(
        cfg=cfg,
        model=model,
        vla_train_dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/space2act_vlm_distill_libero.yaml",
    )
    args, cli_args = parser.parse_known_args()
    cfg = load_config_with_base(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(cli_args)))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    main(cfg)
