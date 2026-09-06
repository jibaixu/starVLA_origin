"""Unit tests for the joint-training causal dynamics bottleneck."""

import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.latent_reasoning import CausalDynamicsBottleneck


class CausalDynamicsBottleneckTests(unittest.TestCase):
    def test_joint_config_uses_plan_before_future_and_declares_cdb(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "examples/simBenchmarks/LIBERO/train_files/space2act_libero_state.yaml"
        )
        config = OmegaConf.load(config_path)
        latent = config.framework.latent_reasoning
        self.assertEqual(latent.token_order, "spatial_instruction_future_plan")
        self.assertEqual(latent.dynamics_bottleneck.mode, "hybrid")
        self.assertEqual(latent.dynamics_bottleneck.dim, 256)
        self.assertFalse(latent.dynamics_bottleneck.enabled)

    def test_cdb_training_entry_has_explicit_cdb_contract_without_default_expansion(self):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "examples/simBenchmarks/LIBERO/train_files/"
            "run_space2act_cdb_confidence_aware_gradnorm_pcgrad_train.sh"
        )
        script = script_path.read_text(encoding="utf-8")
        self.assertNotIn(":-", script)
        self.assertIn("--framework.latent_reasoning.token_order spatial_instruction_plan_future", script)
        self.assertIn("--framework.latent_reasoning.dynamics_bottleneck.enabled", script)
        self.assertIn("--framework.latent_reasoning.dynamics_bottleneck.mode", script)
        self.assertIn("--framework.latent_reasoning.confidence_aware_alignment.enabled true", script)
        self.assertIn("--framework.latent_reasoning.loss_schedule.mode gradnorm", script)
        self.assertIn("--framework.latent_reasoning.pcgrad.enabled true", script)

    def _hidden(self, batch=4, tokens=32, hidden=16):
        torch.manual_seed(7)
        return {
            "spatial": torch.randn(batch, tokens, hidden, requires_grad=True),
            "plan": torch.randn(batch, tokens, hidden, requires_grad=True),
            "future": torch.randn(batch, tokens, hidden, requires_grad=True),
        }

    def test_hybrid_returns_finite_losses_and_backpropagates(self):
        module = CausalDynamicsBottleneck(
            hidden_dim=16,
            state_dim=8,
            dynamics_dim=12,
            mode="hybrid",
        )
        hidden = self._hidden(hidden=16)
        state = torch.randn(4, 1, 8)
        output = module(hidden, state)

        for name in ("loss", "residual_loss", "structural_loss", "intervention_loss", "variance_loss"):
            self.assertTrue(torch.isfinite(output[name]).item(), name)
        self.assertEqual(tuple(output["predicted_future"].shape), (4, 12))
        output["loss"].backward()
        self.assertIsNotNone(hidden["spatial"].grad)
        self.assertIsNotNone(hidden["plan"].grad)
        self.assertIsNotNone(hidden["future"].grad)

    def test_residual_mode_skips_inverse_and_intervention(self):
        module = CausalDynamicsBottleneck(
            hidden_dim=8,
            state_dim=0,
            dynamics_dim=6,
            mode="residual",
        )
        output = module(self._hidden(hidden=8), state=None)
        self.assertEqual(output["structural_loss"].item(), 0.0)
        self.assertEqual(output["intervention_loss"].item(), 0.0)
        self.assertEqual(output["intervention_margin"].item(), 0.0)

    def test_singleton_batch_has_no_invalid_intervention(self):
        module = CausalDynamicsBottleneck(
            hidden_dim=8,
            state_dim=2,
            dynamics_dim=6,
            mode="hybrid",
        )
        output = module(self._hidden(batch=1, hidden=8), torch.zeros(1, 2))
        self.assertTrue(torch.isfinite(output["loss"]).item())
        self.assertEqual(output["intervention_loss"].item(), 0.0)

    def test_rejects_missing_channel_and_wrong_state_shape(self):
        module = CausalDynamicsBottleneck(hidden_dim=8, state_dim=2, dynamics_dim=6)
        hidden = self._hidden(hidden=8)
        with self.assertRaisesRegex(ValueError, "exactly spatial/plan/future"):
            module({"spatial": hidden["spatial"], "plan": hidden["plan"]})
        with self.assertRaisesRegex(ValueError, "state must be"):
            module(hidden, torch.zeros(2, 3))


if __name__ == "__main__":
    unittest.main()
