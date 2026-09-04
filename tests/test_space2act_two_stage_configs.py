"""Static contracts for the Space2Act VLM/action-head split configs."""

from pathlib import Path
import unittest

from omegaconf import OmegaConf

from starVLA.model.framework.share_tools import load_config_with_base
from starVLA.model.framework.VLM4A.space2act_two_stage import normalize_gradnorm_config


class Space2ActTwoStageConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _load(self, filename):
        return load_config_with_base(
            str(self.repo_root / "examples/simBenchmarks/LIBERO/train_files" / filename)
        )

    def test_vlm_distillation_config_trains_all_typed_channels(self):
        source = (
            self.repo_root
            / "examples/simBenchmarks/LIBERO/train_files/space2act_vlm_distill_libero.yaml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("base_config:", source)
        cfg = self._load("space2act_vlm_distill_libero.yaml")
        self.assertEqual(cfg.framework.name, "QwenSpace2ActVLM")
        self.assertNotIn("base_config", cfg)
        self.assertEqual(cfg.framework.latent_reasoning.token_order, "spatial_instruction_plan_future")
        for channel in cfg.framework.latent_reasoning.channels.values():
            self.assertTrue(channel.token_enabled)
            self.assertTrue(channel.alignment_enabled)
        self.assertEqual(cfg.trainer.freeze_modules, "")
        self.assertNotIn("action_model", cfg.framework)
        self.assertNotIn("action_model", cfg.trainer.learning_rate)
        self.assertNotIn("pcgrad", cfg.framework.latent_reasoning)
        self.assertNotIn("evidence_routing", cfg.framework.latent_reasoning)
        self.assertEqual(cfg.framework.latent_reasoning.loss_schedule.mode, "fixed")
        self.assertFalse(cfg.framework.latent_reasoning.loss_schedule.enabled)

    def test_action_head_config_freezes_only_the_transferred_vlm(self):
        source = (
            self.repo_root
            / "examples/simBenchmarks/LIBERO/train_files/space2act_action_head_libero.yaml"
        ).read_text(encoding="utf-8")
        self.assertNotIn("base_config:", source)
        cfg = self._load("space2act_action_head_libero.yaml")
        self.assertEqual(cfg.framework.name, "QwenSpace2ActActionHead")
        self.assertNotIn("base_config", cfg)
        self.assertEqual(cfg.trainer.reload_modules, "qwen_vl_interface")
        self.assertEqual(cfg.trainer.freeze_modules, "qwen_vl_interface")
        self.assertEqual(cfg.framework.action_model.state_dim, 8)
        self.assertEqual(cfg.trainer.learning_rate.action_model, 1.0e-04)
        for channel in cfg.framework.latent_reasoning.channels.values():
            self.assertTrue(channel.token_enabled)
            self.assertFalse(channel.alignment_enabled)
        self.assertNotIn("pcgrad", cfg.framework.latent_reasoning)
        self.assertNotIn("evidence_routing", cfg.framework.latent_reasoning)
        self.assertNotIn("loss_schedule", cfg.framework.latent_reasoning)
        self.assertNotIn("teacher_latents", cfg.datasets.vla_data)

    def test_action_head_cluster_script_uses_only_the_transferred_vlm(self):
        source = (
            self.repo_root
            / "examples/simBenchmarks/LIBERO/train_files/run_space2act_action_head.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("module load cuda/12.4", source)
        self.assertIn("QwenSpace2ActActionHead", source)
        self.assertIn("--trainer.reload_modules qwen_vl_interface", source)
        self.assertIn("--trainer.freeze_modules qwen_vl_interface", source)
        self.assertNotIn(":-", source)

    def test_vlm_config_defaults_to_fixed_without_optional_innovations(self):
        cfg = self._load("space2act_vlm_distill_libero.yaml")
        schedule = cfg.framework.latent_reasoning.loss_schedule
        self.assertFalse(schedule.enabled)
        self.assertEqual(schedule.mode, "fixed")
        self.assertEqual(schedule.knots, [])
        self.assertFalse(cfg.framework.latent_reasoning.confidence_aware_alignment.enabled)
        self.assertNotIn("adaptive_weight_ramp_steps", schedule)

    def test_vlm_innovations_are_script_overrides_of_the_single_yaml(self):
        train_dir = self.repo_root / "examples/simBenchmarks/LIBERO/train_files"
        scripts = (
            "run_space2act_vlm_distill.sh",
            "run_space2act_vlm_progressive.sh",
            "run_space2act_vlm_gradnorm_ramp.sh",
            "run_space2act_vlm_confidence_aware_gradnorm_ramp.sh",
        )
        for filename in scripts:
            source = (train_dir / filename).read_text(encoding="utf-8")
            self.assertIn("module load cuda/12.4", source)
            self.assertIn("train_space2act_vlm.py", source)
            self.assertIn("space2act_vlm_distill_libero.yaml", source)
            self.assertNotIn(":-", source)
            self.assertNotIn("space2act_vlm_progressive_libero.yaml", source)
            self.assertNotIn("space2act_vlm_gradnorm_libero.yaml", source)

    def test_vlm_gradnorm_defaults_match_the_fixed_three_channel_scale(self):
        config = normalize_gradnorm_config(
            {"enabled": True, "mode": "gradnorm"},
            ("spatial", "future", "plan"),
        )
        self.assertEqual(config["auxiliary_budget"], 3.0)
        self.assertEqual(config["min_weight"], 0.25)
        self.assertEqual(config["max_weight"], 2.0)

    def test_vlm_gradnorm_cluster_scripts_use_the_vlm_only_scale(self):
        train_dir = self.repo_root / "examples/simBenchmarks/LIBERO/train_files"
        for filename in (
            "run_space2act_vlm_gradnorm_ramp.sh",
            "run_space2act_vlm_confidence_aware_gradnorm_ramp.sh",
        ):
            source = (train_dir / filename).read_text(encoding="utf-8")
            self.assertIn('LOSS_GRADNORM_AUXILIARY_BUDGET="3.0"', source)
            self.assertIn('LOSS_GRADNORM_MIN_WEIGHT="0.25"', source)
            self.assertIn('LOSS_GRADNORM_MAX_WEIGHT="2.0"', source)

    def test_vlm_progressive_knots_match_the_fixed_three_channel_scale(self):
        source = (
            self.repo_root
            / "examples/simBenchmarks/LIBERO/train_files/run_space2act_vlm_progressive.sh"
        ).read_text(encoding="utf-8")
        raw_knots = source.split("--framework.latent_reasoning.loss_schedule.knots '", 1)[1].split("'\n", 1)[0]
        knots = OmegaConf.create(raw_knots)
        for knot in knots:
            self.assertAlmostEqual(sum(knot.weights.values()), 3.0)

    def test_vlm_stage_keeps_both_teacher_storage_backends_configurable(self):
        cfg = self._load("space2act_vlm_distill_libero.yaml")
        for channel in cfg.datasets.vla_data.teacher_latents.channels.values():
            self.assertIn("relative_dir", channel)
            self.assertIn("packed_relative_dir", channel)

    def test_vlm_framework_does_not_depend_on_end_to_end_framework(self):
        source = (
            self.repo_root / "starVLA/model/framework/VLM4A/QwenSpace2ActVLM.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("QwenSpace2ActGR00T import", source)
        self.assertNotIn("QwenSpace2Act_GR00T", source)

    def test_action_head_framework_excludes_distillation_components(self):
        source = (
            self.repo_root / "starVLA/model/framework/VLM4A/QwenSpace2ActActionHead.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "CrossAttentionLatentAligner",
            "ActionGuidedEvidenceRouter",
            "pcgrad",
            "teacher_latents",
            "confidence_aware_alignment",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
