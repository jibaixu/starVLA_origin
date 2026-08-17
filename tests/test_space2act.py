"""Space2Act 教师、对齐映射、checkpoint 与采样 seed 的核心单元测试。"""

import ast
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from starVLA.dataloader.gr00t_lerobot.teacher_latents import (
    EpisodeTeacherStore,
    TeacherChannelSpec,
)
from starVLA.model.modules.latent_reasoning import CrossAttentionLatentAligner
from starVLA.model.framework.VLM4A.QwenSpace2ActGR00T import (
    QwenSpace2Act_GR00T,
    _normalize_dwa_config,
    _interpolate_loss_schedule,
    _normalize_loss_schedule,
)
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset


def _write_teacher(
    root: Path,
    features: torch.Tensor,
    anchors: torch.Tensor,
    metadata: dict[str, str] | None = None,
) -> None:
    """按真实目录和字段约定生成一个最小 safetensors 教师文件。"""
    path = root / "latents" / "chunk-000" / "episode_000000.safetensors"
    path.parent.mkdir(parents=True)
    save_file(
        {"features": features, "anchor_frames": anchors},
        path,
        metadata=metadata,
    )


class Space2ActTest(unittest.TestCase):
    """覆盖无需加载完整 Qwen/GR00T 权重即可验证的关键边界行为。"""

    def test_teacher_store_uses_latest_previous_anchor(self):
        """稀疏教师应复用不晚于当前帧的最近 anchor latent。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            features = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
            _write_teacher(root, features, torch.tensor([0, 16, 32]))
            store = EpisodeTeacherStore(
                root, TeacherChannelSpec("spatial", "latents", 2, 4), [0]
            )

            self.assertTrue(torch.equal(store.get(0, 0), features[0]))
            self.assertTrue(torch.equal(store.get(0, 15), features[0]))
            self.assertTrue(torch.equal(store.get(0, 16), features[1]))
            self.assertTrue(torch.equal(store.get(0, 100), features[2]))

    def test_teacher_store_fails_on_shape_mismatch(self):
        """教师张量与配置契约不一致时，应在建索引阶段尽早报错。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_teacher(root, torch.zeros(2, 3, 4), torch.tensor([0, 16]))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                EpisodeTeacherStore(
                    root, TeacherChannelSpec("spatial", "latents", 2, 4), [0]
                )

    @staticmethod
    def _future_spec() -> TeacherChannelSpec:
        return TeacherChannelSpec(
            "future",
            "latents",
            2,
            4,
            feature_dtype="bfloat16",
            anchor_dtype="int64",
            anchor_stride=8,
            validate_episode_length=True,
            expected_metadata={"complete": "true", "share_stride": "8"},
        )

    @staticmethod
    def _future_metadata(**overrides) -> dict[str, str]:
        metadata = {
            "complete": "true",
            "share_stride": "8",
            "suite": "libero_goal",
            "episode_index": "0",
            "episode_length": "17",
        }
        metadata.update({key: str(value) for key, value in overrides.items()})
        return metadata

    def test_future_teacher_contract_and_stride_mapping(self):
        """Stage 2 teacher 应校验 BF16/stride/episode，并按 latest anchor 共享。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "libero_goal_no_noops_1.0.0_lerobot"
            features = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4).to(
                torch.bfloat16
            )
            _write_teacher(
                root,
                features,
                torch.tensor([0, 8, 16], dtype=torch.int64),
                self._future_metadata(),
            )
            store = EpisodeTeacherStore(
                root,
                self._future_spec(),
                [0],
                episode_lengths={0: 17},
            )

            self.assertTrue(torch.equal(store.get(0, 0), features[0]))
            self.assertTrue(torch.equal(store.get(0, 7), features[0]))
            self.assertTrue(torch.equal(store.get(0, 8), features[1]))
            self.assertTrue(torch.equal(store.get(0, 16), features[2]))

    def test_future_teacher_rejects_dtype_anchor_and_metadata_mismatch(self):
        """错误精度、anchor 起点/stride 或 episode metadata 必须在训练前失败。"""
        cases = (
            (
                "feature dtype mismatch",
                torch.zeros(3, 2, 4, dtype=torch.float32),
                torch.tensor([0, 8, 16], dtype=torch.int64),
                self._future_metadata(),
            ),
            (
                "start at frame 0",
                torch.zeros(3, 2, 4, dtype=torch.bfloat16),
                torch.tensor([1, 9, 16], dtype=torch.int64),
                self._future_metadata(),
            ),
            (
                "stride mismatch",
                torch.zeros(3, 2, 4, dtype=torch.bfloat16),
                torch.tensor([0, 7, 16], dtype=torch.int64),
                self._future_metadata(),
            ),
            (
                "episode_length mismatch",
                torch.zeros(3, 2, 4, dtype=torch.bfloat16),
                torch.tensor([0, 8, 16], dtype=torch.int64),
                self._future_metadata(episode_length=18),
            ),
        )
        for error, features, anchors, metadata in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir) / "libero_goal_no_noops_1.0.0_lerobot"
                _write_teacher(root, features, anchors, metadata)
                with self.assertRaisesRegex(ValueError, error):
                    EpisodeTeacherStore(
                        root,
                        self._future_spec(),
                        [0],
                        episode_lengths={0: 17},
                    )

    @staticmethod
    def _plan_spec() -> TeacherChannelSpec:
        return TeacherChannelSpec(
            "plan",
            "latents",
            32,
            32,
            feature_dtype="bfloat16",
            anchor_dtype="int64",
            anchor_stride=1,
            validate_episode_length=True,
            expected_metadata={
                "complete": "true",
                "checkpoint_epoch": "10",
                "checkpoint_sha256": (
                    "4d48688c53c8830da6c2f4899264771f8c4c00595d1462794964589efa09ff46"
                ),
                "segment_offsets": "0:4,4:8,8:12,12:16",
                "latent_type": "pre_vq",
            },
        )

    @staticmethod
    def _plan_metadata(**overrides) -> dict[str, str]:
        metadata = {
            "complete": "true",
            "checkpoint_epoch": "10",
            "checkpoint_sha256": (
                "4d48688c53c8830da6c2f4899264771f8c4c00595d1462794964589efa09ff46"
            ),
            "segment_offsets": "0:4,4:8,8:12,12:16",
            "latent_type": "pre_vq",
            "suite": "libero_goal",
            "episode_index": "0",
            "episode_length": "17",
        }
        metadata.update({key: str(value) for key, value in overrides.items()})
        return metadata

    def test_plan_teacher_contract_maps_every_frame(self):
        """Stage 3 Moto teacher 应逐帧提供四段拼接后的 [32,32] pre-VQ latent。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "libero_goal_no_noops_1.0.0_lerobot"
            features = torch.arange(17 * 32 * 32, dtype=torch.float32).reshape(17, 32, 32)
            features = features.to(torch.bfloat16)
            _write_teacher(
                root,
                features,
                torch.arange(17, dtype=torch.int64),
                self._plan_metadata(),
            )
            store = EpisodeTeacherStore(
                root,
                self._plan_spec(),
                [0],
                episode_lengths={0: 17},
            )

            self.assertTrue(torch.equal(store.get(0, 0), features[0]))
            self.assertTrue(torch.equal(store.get(0, 9), features[9]))
            self.assertTrue(torch.equal(store.get(0, 16), features[16]))

    def test_plan_teacher_rejects_wrong_checkpoint_or_stride(self):
        """Moto checkpoint 身份或逐帧 anchor 契约改变时，Stage 3 必须拒绝启动。"""
        cases = (
            (
                "checkpoint_sha256",
                torch.arange(17, dtype=torch.int64),
                self._plan_metadata(checkpoint_sha256="wrong"),
            ),
            (
                "stride mismatch",
                torch.arange(0, 34, 2, dtype=torch.int64),
                self._plan_metadata(),
            ),
        )
        for error, anchors, metadata in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir) / "libero_goal_no_noops_1.0.0_lerobot"
                _write_teacher(
                    root,
                    torch.zeros(17, 32, 32, dtype=torch.bfloat16),
                    anchors,
                    metadata,
                )
                with self.assertRaisesRegex(ValueError, error):
                    EpisodeTeacherStore(
                        root,
                        self._plan_spec(),
                        [0],
                        episode_lengths={0: 17},
                    )

    def test_chunked_aligner_matches_unchunked_and_backpropagates(self):
        """分块 cross-attention 应与完整计算等价，且梯度可回传到输入和 query。"""
        torch.manual_seed(0)
        aligner = CrossAttentionLatentAligner(
            input_dim=8,
            target_tokens=7,
            target_dim=6,
            alignment_dim=8,
            num_heads=2,
            query_chunk_size=3,
        )
        hidden = torch.randn(2, 4, 8, requires_grad=True)
        teachers = [torch.randn(7, 6), torch.randn(7, 6)]

        chunked_loss = aligner.compute_loss(hidden, teachers)
        aligner.query_chunk_size = 7
        full_loss = aligner.compute_loss(hidden, teachers)
        self.assertTrue(torch.allclose(chunked_loss, full_loss, atol=1e-6, rtol=1e-5))

        chunked_loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertIsNotNone(aligner.target_queries.grad)

    def test_aligner_rejects_wrong_teacher_shape(self):
        """对齐损失不得接受 token 数或维度错误的教师样本。"""
        aligner = CrossAttentionLatentAligner(8, 7, 6, alignment_dim=8, num_heads=2)
        with self.assertRaisesRegex(ValueError, "Expected teacher shape"):
            aligner.compute_loss(torch.randn(1, 4, 8), [torch.randn(6, 6)])

    def test_progressive_schedule_interpolates_fixed_auxiliary_budget(self):
        """Progressive knots 应平滑交叉渐变，并在 warmup 后保持 0.30 总预算。"""
        repo_root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(
            repo_root / "examples/simBenchmarks/LIBERO/train_files/space2act_libero_state.yaml"
        )
        knots = _normalize_loss_schedule(
            OmegaConf.merge(cfg.framework.latent_reasoning.loss_schedule, {"enabled": True}),
            {"spatial", "future", "plan"},
        )

        expected = {
            0: (0.0, 0.0, 0.0),
            2500: (0.15, 0.0, 0.0),
            5000: (0.30, 0.0, 0.0),
            22500: (0.20, 0.10, 0.0),
            52500: (0.075, 0.15, 0.075),
            100000: (0.05, 0.10, 0.15),
            120000: (0.05, 0.10, 0.15),
        }
        for step, values in expected.items():
            with self.subTest(step=step):
                weights = _interpolate_loss_schedule(knots, step)
                actual = tuple(weights[channel] for channel in ("spatial", "future", "plan"))
                self.assertTrue(np.allclose(actual, values), (step, actual))
                if step >= 5000:
                    self.assertAlmostEqual(sum(actual), 0.30)

    def test_progressive_schedule_validation_rejects_ambiguous_contracts(self):
        """Progressive 模式必须显式启用三通道，并使用完整、单调、非负 knots。"""
        valid = {
            "enabled": True,
            "knots": [
                {
                    "step": 0,
                    "weights": {"spatial": 0.3, "future": 0.0, "plan": 0.0},
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "requires all"):
            _normalize_loss_schedule(valid, {"spatial", "future"})

        invalid_cases = (
            (
                "strictly increasing",
                [valid["knots"][0], valid["knots"][0]],
            ),
            (
                "finite and non-negative",
                [
                    {
                        "step": 0,
                        "weights": {"spatial": -0.1, "future": 0.2, "plan": 0.2},
                    }
                ],
            ),
            (
                "exactly spatial/future/plan",
                [{"step": 0, "weights": {"spatial": 0.3, "future": 0.0}}],
            ),
        )
        for error, knots in invalid_cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                _normalize_loss_schedule(
                    {"enabled": True, "knots": knots},
                    {"spatial", "future", "plan"},
                )

    def test_legacy_weights_ignore_step_and_progressive_requires_it(self):
        """旧配置必须保持固定权重；progressive 则不得在缺少 global step 时静默运行。"""
        framework = object.__new__(QwenSpace2Act_GR00T)
        framework.loss_schedule_enabled = False
        framework.alignment_loss_weights = {"spatial": 1.0, "future": 0.3}
        self.assertEqual(
            QwenSpace2Act_GR00T._effective_alignment_weights(framework, 52500),
            {"spatial": 1.0, "future": 0.3},
        )

        framework.loss_schedule_enabled = True
        framework.loss_schedule_knots = (
            (0, {"spatial": 0.0, "future": 0.0, "plan": 0.0}),
            (10, {"spatial": 0.3, "future": 0.0, "plan": 0.0}),
        )
        with self.assertRaisesRegex(ValueError, "training_step is required"):
            QwenSpace2Act_GR00T._effective_alignment_weights(framework, None)

    def test_dwa_warmup_and_weight_budget(self):
        """DWA uses a zero auxiliary budget during warmup and preserves the budget after it."""
        config = _normalize_dwa_config(
            {
                "mode": "dwa",
                "enabled": True,
                "auxiliary_budget": 0.3,
                "warmup_steps": 2,
                "update_interval": 1,
                "temperature": 1.0,
                "ema_decay": 0.0,
            },
            {"spatial", "future", "plan"},
        )
        framework = object.__new__(QwenSpace2Act_GR00T)
        framework.dwa_enabled = True
        framework.dwa_config = config
        framework.dwa_channels = ("spatial", "future", "plan")
        framework.dwa_weights = {channel: 0.1 for channel in framework.dwa_channels}
        framework.dwa_loss_history = {channel: [] for channel in framework.dwa_channels}
        framework.dwa_loss_ema = {channel: None for channel in framework.dwa_channels}
        framework.dwa_rates = {channel: 1.0 for channel in framework.dwa_channels}

        self.assertEqual(
            QwenSpace2Act_GR00T._effective_alignment_weights(framework, 2),
            {channel: 0.0 for channel in framework.dwa_channels},
        )
        losses = {channel: torch.tensor(value) for channel, value in {
            "spatial": 1.0, "future": 2.0, "plan": 3.0
        }.items()}
        QwenSpace2Act_GR00T.update_adaptive_state(framework, 2, losses)
        QwenSpace2Act_GR00T.update_adaptive_state(framework, 3, losses)
        next_losses = {channel: torch.tensor(value) for channel, value in {
            "spatial": 0.5, "future": 1.5, "plan": 3.0
        }.items()}
        QwenSpace2Act_GR00T.update_adaptive_state(framework, 4, next_losses)
        self.assertAlmostEqual(sum(framework.dwa_weights.values()), 0.3, places=6)
        self.assertGreater(framework.dwa_weights["plan"], framework.dwa_weights["spatial"])

        with self.assertRaisesRegex(ValueError, "cannot define fixed knots"):
            _normalize_dwa_config(
                {"mode": "dwa", "enabled": True, "knots": [{"step": 0}]},
                {"spatial"},
            )

    def test_dwa_state_round_trip(self):
        """DWA runtime state is JSON-safe and restores its weights and history."""
        framework = object.__new__(QwenSpace2Act_GR00T)
        framework.dwa_enabled = True
        framework.dwa_config = _normalize_dwa_config(
            {"mode": "dwa", "enabled": True, "auxiliary_budget": 0.3},
            {"spatial", "future"},
        )
        framework.dwa_channels = ("spatial", "future")
        framework.dwa_weights = {"spatial": 0.11, "future": 0.19}
        framework.dwa_loss_history = {"spatial": [0.4, 0.3], "future": [0.8, 0.7]}
        framework.dwa_loss_ema = {"spatial": 0.3, "future": 0.7}
        framework.dwa_rates = {"spatial": 0.75, "future": 0.875}
        state = QwenSpace2Act_GR00T.get_adaptive_state(framework)

        restored = object.__new__(QwenSpace2Act_GR00T)
        restored.dwa_enabled = True
        restored.dwa_config = framework.dwa_config
        restored.dwa_channels = framework.dwa_channels
        restored.dwa_weights = {channel: 0.15 for channel in restored.dwa_channels}
        restored.dwa_loss_history = {channel: [] for channel in restored.dwa_channels}
        restored.dwa_loss_ema = {channel: None for channel in restored.dwa_channels}
        restored.dwa_rates = {channel: 1.0 for channel in restored.dwa_channels}
        QwenSpace2Act_GR00T.set_adaptive_state(restored, state)
        self.assertEqual(restored.dwa_loss_history, framework.dwa_loss_history)
        self.assertEqual(restored.dwa_weights, framework.dwa_weights)

    def test_hidden_gradient_diagnostics_do_not_accumulate_parameter_grads(self):
        """诊断使用 autograd.grad 读取 hidden 梯度，不应提前写入正常 backward 梯度。"""
        framework = object.__new__(QwenSpace2Act_GR00T)
        hidden = torch.randn(1, 6, 4, requires_grad=True)
        action_loss = hidden.square().mean()
        channel_losses = {
            "spatial": hidden[:, 1:3].square().mean(),
            "future": hidden[:, 3:5].square().mean(),
        }
        positions = {
            "spatial": torch.tensor([[1, 2]]),
            "future": torch.tensor([[3, 4]]),
        }
        weights = {"spatial": 0.1, "future": 0.2}

        diagnostics = QwenSpace2Act_GR00T._compute_gradient_diagnostics(
            framework,
            hidden,
            action_loss,
            channel_losses,
            positions,
            weights,
        )
        self.assertIsNone(hidden.grad)
        self.assertTrue(all(torch.isfinite(value) for value in diagnostics.values()))
        self.assertGreater(diagnostics["gradient_cosine/spatial_vs_action"].item(), 0.0)

        total_loss = action_loss + sum(weights[name] * loss for name, loss in channel_losses.items())
        total_loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_base_checkpoint_keeps_new_token_initialization(self):
        """加载旧 base 权重时，新增 Space2Act token 行应保留当前初始化值。"""
        key = "qwen_vl_interface.model.model.language_model.embed_tokens.weight"
        current_weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)

        class TokenInterface:
            space2act_token_ids = {"spatial": [3], "future": [], "plan": []}

        class FakeFramework:
            qwen_vl_interface = TokenInterface()

            def state_dict(self):
                return {key: current_weight}

        checkpoint = {key: torch.ones_like(current_weight)}
        adapted = QwenSpace2Act_GR00T.adapt_checkpoint_state_dict(
            FakeFramework(), checkpoint
        )
        self.assertTrue(torch.equal(adapted[key][0], checkpoint[key][0]))
        self.assertTrue(torch.equal(adapted[key][3], current_weight[3]))

    def test_stage2_checkpoint_adds_only_future_mapper(self):
        """Stage 1 热启动应恢复已有参数，并只保留新 future mapper 的初始化。"""
        current = {
            "backbone.weight": torch.tensor([1.0]),
            "alignment_heads.spatial.weight": torch.tensor([2.0]),
            "alignment_heads.future.weight": torch.tensor([3.0]),
        }

        class TokenInterface:
            space2act_token_ids = {"spatial": [], "future": [], "plan": []}

        class FakeFramework:
            space2act_stage = 2
            qwen_vl_interface = TokenInterface()
            alignment_heads = {"spatial": object(), "future": object()}

            def state_dict(self):
                return current

        checkpoint = {
            "backbone.weight": torch.tensor([10.0]),
            "alignment_heads.spatial.weight": torch.tensor([20.0]),
        }
        adapted = QwenSpace2Act_GR00T.adapt_checkpoint_state_dict(FakeFramework(), checkpoint)
        self.assertTrue(torch.equal(adapted["backbone.weight"], checkpoint["backbone.weight"]))
        self.assertTrue(
            torch.equal(
                adapted["alignment_heads.spatial.weight"],
                checkpoint["alignment_heads.spatial.weight"],
            )
        )
        self.assertTrue(
            torch.equal(adapted["alignment_heads.future.weight"], current["alignment_heads.future.weight"])
        )

    def test_stage2_rejects_non_space2act_checkpoint(self):
        """Stage 2 不允许绕过课程学习直接加载普通 base checkpoint。"""

        class TokenInterface:
            space2act_token_ids = {"spatial": [], "future": [], "plan": []}

        class FakeFramework:
            space2act_stage = 2
            qwen_vl_interface = TokenInterface()
            alignment_heads = {"spatial": object(), "future": object()}

            def state_dict(self):
                return {"backbone.weight": torch.tensor([1.0])}

        with self.assertRaisesRegex(ValueError, "must warm-start"):
            QwenSpace2Act_GR00T.adapt_checkpoint_state_dict(
                FakeFramework(), {"backbone.weight": torch.tensor([2.0])}
            )

    def test_stage3_checkpoint_adds_only_plan_mapper(self):
        """Stage 2 热启动应恢复 S/F，并只保留新 plan mapper 的初始化。"""
        current = {
            "backbone.weight": torch.tensor([1.0]),
            "alignment_heads.spatial.weight": torch.tensor([2.0]),
            "alignment_heads.future.weight": torch.tensor([3.0]),
            "alignment_heads.plan.weight": torch.tensor([4.0]),
        }

        class TokenInterface:
            space2act_token_ids = {"spatial": [], "future": [], "plan": []}

        class FakeFramework:
            space2act_stage = 3
            qwen_vl_interface = TokenInterface()
            alignment_heads = {"spatial": object(), "future": object(), "plan": object()}

            def state_dict(self):
                return current

        checkpoint = {
            "backbone.weight": torch.tensor([10.0]),
            "alignment_heads.spatial.weight": torch.tensor([20.0]),
            "alignment_heads.future.weight": torch.tensor([30.0]),
        }
        adapted = QwenSpace2Act_GR00T.adapt_checkpoint_state_dict(FakeFramework(), checkpoint)
        self.assertTrue(torch.equal(adapted["backbone.weight"], checkpoint["backbone.weight"]))
        self.assertTrue(
            torch.equal(
                adapted["alignment_heads.spatial.weight"],
                checkpoint["alignment_heads.spatial.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                adapted["alignment_heads.future.weight"],
                checkpoint["alignment_heads.future.weight"],
            )
        )
        self.assertTrue(
            torch.equal(adapted["alignment_heads.plan.weight"], current["alignment_heads.plan.weight"])
        )

    def test_stage3_same_stage_checkpoint_resumes_all_mappers(self):
        """Stage 3 restart/resume 不应重置已经训练过的 plan mapper。"""
        current = {
            "backbone.weight": torch.tensor([1.0]),
            "alignment_heads.spatial.weight": torch.tensor([2.0]),
            "alignment_heads.future.weight": torch.tensor([3.0]),
            "alignment_heads.plan.weight": torch.tensor([4.0]),
        }

        class TokenInterface:
            space2act_token_ids = {"spatial": [], "future": [], "plan": []}

        class FakeFramework:
            space2act_stage = 3
            qwen_vl_interface = TokenInterface()
            alignment_heads = {"spatial": object(), "future": object(), "plan": object()}

            def state_dict(self):
                return current

        checkpoint = {key: value + 10 for key, value in current.items()}
        adapted = QwenSpace2Act_GR00T.adapt_checkpoint_state_dict(FakeFramework(), checkpoint)
        for key in current:
            self.assertTrue(torch.equal(adapted[key], checkpoint[key]))

    def test_stage3_rejects_stage1_checkpoint(self):
        """Stage 3 不允许跳过 future 课程，直接从只有 spatial mapper 的 Stage 1 启动。"""

        class TokenInterface:
            space2act_token_ids = {"spatial": [], "future": [], "plan": []}

        class FakeFramework:
            space2act_stage = 3
            qwen_vl_interface = TokenInterface()
            alignment_heads = {"spatial": object(), "future": object(), "plan": object()}

            def state_dict(self):
                return {
                    "backbone.weight": torch.tensor([1.0]),
                    "alignment_heads.spatial.weight": torch.tensor([2.0]),
                    "alignment_heads.future.weight": torch.tensor([3.0]),
                    "alignment_heads.plan.weight": torch.tensor([4.0]),
                }

        with self.assertRaisesRegex(ValueError, "predecessor checkpoint"):
            QwenSpace2Act_GR00T.adapt_checkpoint_state_dict(
                FakeFramework(),
                {
                    "backbone.weight": torch.tensor([10.0]),
                    "alignment_heads.spatial.weight": torch.tensor([20.0]),
                },
            )

    def test_stage3_yaml_and_shell_contract(self):
        """Stage 3 配置应锁定三教师契约，并保持 shell 顶部直接赋默认值。"""
        repo_root = Path(__file__).resolve().parents[1]
        config_path = (
            repo_root
            / "examples/simBenchmarks/LIBERO/train_files/space2act_libero_state.yaml"
        )
        cfg = OmegaConf.load(config_path)
        future = cfg.framework.latent_reasoning.channels.future
        teacher = cfg.datasets.vla_data.teacher_latents.channels.future
        self.assertEqual((future.target_tokens, future.target_dim), (1125, 3072))
        self.assertEqual(teacher.anchor_stride, 8)
        self.assertEqual(teacher.expected_metadata.block_index, "20")
        self.assertEqual(teacher.expected_metadata.saved_denoise_step, "19")

        plan = cfg.framework.latent_reasoning.channels.plan
        plan_teacher = cfg.datasets.vla_data.teacher_latents.channels.plan
        self.assertEqual((plan.target_tokens, plan.target_dim), (32, 32))
        self.assertEqual(plan_teacher.anchor_stride, 1)
        self.assertEqual(plan_teacher.feature_dtype, "bfloat16")
        self.assertEqual(plan_teacher.expected_metadata.checkpoint_epoch, "10")
        self.assertEqual(
            plan_teacher.expected_metadata.checkpoint_sha256,
            "4d48688c53c8830da6c2f4899264771f8c4c00595d1462794964589efa09ff46",
        )
        self.assertEqual(
            plan_teacher.expected_metadata.segment_offsets,
            "0:4,4:8,8:12,12:16",
        )

        shell_paths = (
            repo_root / "examples/simBenchmarks/LIBERO/train_files/run_space2act_train.sh",
            repo_root
            / "examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/run_auto_eval_libero.sh",
            repo_root
            / "examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall_jbx.sh",
        )
        default_expansion = re.compile(r"\$\{[^}]*:-")
        for shell_path in shell_paths:
            text = shell_path.read_text()
            self.assertIsNone(default_expansion.search(text), shell_path)

        progressive_path = (
            repo_root
            / "examples/simBenchmarks/LIBERO/train_files/run_space2act_progressive_train.sh"
        )
        self.assertIsNone(default_expansion.search(progressive_path.read_text()), progressive_path)

        train_text = shell_paths[0].read_text()
        self.assertIn('SPACE2ACT_STAGE="3"', train_text)
        self.assertIn('INIT_CHECKPOINT=""', train_text)
        self.assertIn('RUN_ID="space2act_libero_stage${SPACE2ACT_STAGE}_cold_start"', train_text)
        self.assertIn('stage_tokens="spatial,future,plan"', train_text)
        self.assertIn("teacher_latents.channels.plan.relative_dir", train_text)
        self.assertNotIn("Stage 3 is not implemented", train_text)
        self.assertNotIn("requires INIT_CHECKPOINT to point", train_text)

        eval_text = shell_paths[1].read_text()
        self.assertIn("space2act_libero_stage3", eval_text)
        self.assertIn("steps_100000_pytorch_model.pt", eval_text)

    def test_training_shell_supports_cold_start_and_strict_warm_start(self):
        """空值应 cold-start；非空 checkpoint 应传入命令，且坏路径须提前失败。"""
        repo_root = Path(__file__).resolve().parents[1]
        source = (
            repo_root / "examples/simBenchmarks/LIBERO/train_files/run_space2act_train.sh"
        ).read_text()
        dry_run_source = source.replace('DRY_RUN="0"', 'DRY_RUN="1"')

        with tempfile.TemporaryDirectory() as temp_dir:
            cold_script = Path(temp_dir) / "cold_start.sh"
            cold_script.write_text(dry_run_source)
            cold = subprocess.run(
                ["bash", str(cold_script)],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(cold.returncode, 0, cold.stderr)
            self.assertIn("space2act_libero_stage3_cold_start", cold.stdout)
            self.assertNotIn("--trainer.pretrained_checkpoint", cold.stdout)

            for stage, expected_run_id in (
                (1, "space2act_libero_stage1"),
                (2, "space2act_libero_stage2_cold_start"),
            ):
                stage_script = Path(temp_dir) / f"stage{stage}.sh"
                stage_script.write_text(
                    dry_run_source.replace('SPACE2ACT_STAGE="3"', f'SPACE2ACT_STAGE="{stage}"')
                )
                stage_run = subprocess.run(
                    ["bash", str(stage_script)],
                    cwd=repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(stage_run.returncode, 0, stage_run.stderr)
                self.assertIn(f"--run_id {expected_run_id}", stage_run.stdout)

            warm_checkpoint = Path(temp_dir) / "stage2.pt"
            warm_checkpoint.touch()
            warm_script = Path(temp_dir) / "warm_start.sh"
            warm_script.write_text(
                dry_run_source.replace(
                    'INIT_CHECKPOINT=""',
                    f'INIT_CHECKPOINT="{warm_checkpoint}"',
                )
            )
            warm = subprocess.run(
                ["bash", str(warm_script)],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(warm.returncode, 0, warm.stderr)
            self.assertIn("--run_id space2act_libero_stage3", warm.stdout)
            self.assertNotIn("space2act_libero_stage3_cold_start", warm.stdout)
            self.assertIn(
                f"--trainer.pretrained_checkpoint {warm_checkpoint}",
                warm.stdout,
            )

            missing_script = Path(temp_dir) / "missing_checkpoint.sh"
            missing_script.write_text(
                dry_run_source.replace(
                    'INIT_CHECKPOINT=""',
                    'INIT_CHECKPOINT="/definitely/missing/space2act_stage2.pt"',
                )
            )
            missing = subprocess.run(
                ["bash", str(missing_script)],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("INIT_CHECKPOINT does not exist", missing.stderr)

    def test_progressive_training_shell_has_fixed_sequence_contract(self):
        """新入口应单次 cold-start 100k，并始终启用 S/F/P token 和 teacher。"""
        repo_root = Path(__file__).resolve().parents[1]
        path = (
            repo_root
            / "examples/simBenchmarks/LIBERO/train_files/run_space2act_progressive_train.sh"
        )
        source = path.read_text()
        dry_run_source = source.replace('DRY_RUN="0"', 'DRY_RUN="1"')

        with tempfile.TemporaryDirectory() as temp_dir:
            script = Path(temp_dir) / "progressive.sh"
            script.write_text(dry_run_source)
            result = subprocess.run(
                ["bash", str(script)],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--framework.latent_reasoning.stage 3", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.mode fixed", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.enabled true", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.diagnostic_interval 1000", result.stdout)
        self.assertNotIn("dwa", source.lower())
        self.assertNotIn("LOSS_DWA", source)
        self.assertNotIn("--framework.latent_reasoning.loss_schedule.knots", result.stdout)
        for channel in ("spatial", "future", "plan"):
            self.assertIn(
                f"--framework.latent_reasoning.channels.{channel}.token_enabled true",
                result.stdout,
            )
            self.assertIn(
                f"--framework.latent_reasoning.channels.{channel}.alignment_enabled true",
                result.stdout,
            )
        self.assertIn("--trainer.max_train_steps 100000", result.stdout)
        self.assertIn("space2act_libero_progressive_sfp_100k", result.stdout)
        self.assertNotIn("--trainer.pretrained_checkpoint", result.stdout)

        trainer_source = (repo_root / "starVLA/training/train_starvla.py").read_text()
        self.assertIn('forward_kwargs["training_step"] = optimizer_step', trainer_source)
        self.assertIn(
            'forward_kwargs["run_gradient_diagnostics"] = self.accelerator.sync_gradients',
            trainer_source,
        )

    def test_dwa_training_shell_has_dedicated_contract(self):
        """DWA 入口应固定启用自适应模式，并与 progressive knots 完全解耦。"""
        repo_root = Path(__file__).resolve().parents[1]
        path = repo_root / "examples/simBenchmarks/LIBERO/train_files/run_space2act_dwa_train.sh"
        source = path.read_text()
        dry_run_source = source.replace('DRY_RUN="0"', 'DRY_RUN="1"')

        with tempfile.TemporaryDirectory() as temp_dir:
            script = Path(temp_dir) / "dwa.sh"
            script.write_text(dry_run_source)
            result = subprocess.run(
                ["bash", str(script)],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--framework.latent_reasoning.loss_schedule.mode dwa", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.enabled true", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.knots \\[\\]", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.warmup_steps 5000", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.update_interval 100", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.temperature 2.0", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.ema_decay 0.9", result.stdout)
        self.assertIn("--framework.latent_reasoning.loss_schedule.auxiliary_budget 0.30", result.stdout)
        self.assertIn("--run_id space2act_libero_dwa_sfp_100k", result.stdout)
        self.assertNotIn("LOSS_SCHEDULE_MODE", source)
        self.assertNotIn("progressive_sfp", source)

    @staticmethod
    def _debug_option(argv: list[str], name: str) -> str:
        return argv[argv.index(name) + 1]

    @staticmethod
    def _commented_debug_argv(source: str, start_marker: str, end_marker: str) -> list[str]:
        section = source.split(start_marker, 1)[1].split(end_marker, 1)[0]
        uncommented = "\n".join(
            line.replace("    # ", "    ", 1) if line.startswith("    # ") else line
            for line in section.splitlines()
        )
        tree = ast.parse(textwrap.dedent(uncommented))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "argv"
        )
        return ast.literal_eval(assignment.value)

    def test_debug_on_has_runnable_stage_templates(self):
        """debug_on 默认运行 Stage 3，同时保留参数完整的 Stage 1/2 注释模板。"""
        repo_root = Path(__file__).resolve().parents[1]
        trainer_path = repo_root / "starVLA/training/train_starvla.py"
        source = trainer_path.read_text()
        module = ast.parse(source)
        debug_function = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "debug_on"
        )
        namespace = {}
        exec(compile(ast.Module(body=[debug_function], type_ignores=[]), trainer_path, "exec"), namespace)
        with patch("sys.argv", ["train_starvla.py"]), patch.dict("os.environ", {}, clear=False):
            namespace["debug_on"]()
            stage3 = list(__import__("sys").argv)

        stage1 = self._commented_debug_argv(
            source,
            "    # 第一阶段（需要时取消注释，并注释下方第三阶段）",
            "    # 第二阶段（需要时取消注释，并注释下方第三阶段）",
        )
        stage2 = self._commented_debug_argv(
            source,
            "    # 第二阶段（需要时取消注释，并注释下方第三阶段）",
            "    # 第三阶段",
        )
        stages = {1: stage1, 2: stage2, 3: stage3}
        for stage, argv in stages.items():
            with self.subTest(stage=stage):
                self.assertEqual(
                    self._debug_option(argv, "--framework.latent_reasoning.stage"), str(stage)
                )
                self.assertEqual(
                    self._debug_option(argv, "--datasets.vla_data.episode_indices"), "[0,1,2]"
                )
                self.assertEqual(
                    self._debug_option(
                        argv, "--framework.latent_reasoning.channels.spatial.alignment_enabled"
                    ),
                    "true",
                )
                self.assertEqual(
                    self._debug_option(
                        argv, "--framework.latent_reasoning.channels.future.alignment_enabled"
                    ),
                    "true" if stage >= 2 else "false",
                )
                self.assertEqual(
                    self._debug_option(
                        argv, "--framework.latent_reasoning.channels.plan.alignment_enabled"
                    ),
                    "true" if stage == 3 else "false",
                )
                self.assertEqual(
                    self._debug_option(argv, "--run_id"),
                    f"debug_space2act_libero_stage{stage}_ep0_2",
                )

        self.assertNotIn("--trainer.pretrained_checkpoint", stage1)
        self.assertEqual(self._debug_option(stage2, "--trainer.pretrained_checkpoint"), "")
        self.assertEqual(self._debug_option(stage3, "--trainer.pretrained_checkpoint"), "")
        self.assertIn("--datasets.vla_data.teacher_latents.channels.future.relative_dir", stage2)
        self.assertIn("--datasets.vla_data.teacher_latents.channels.plan.relative_dir", stage3)

    @staticmethod
    def _sampling_dataset(seed: int) -> LeRobotMixtureDataset:
        """绕过磁盘数据构造，只保留 sample_step 所需的最小采样状态。"""
        dataset = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
        trajectory_count = 32
        dataset.datasets = [
            SimpleNamespace(
                trajectory_ids=np.arange(trajectory_count),
                trajectory_lengths=np.full(trajectory_count, 64),
            )
        ]
        dataset._dataset_sampling_weights = np.array([1.0])
        dataset._trajectory_sampling_weights = [
            np.full(trajectory_count, 1.0 / trajectory_count)
        ]
        dataset.seed = seed
        dataset.mode = "train"
        dataset.set_epoch(0)
        return dataset

    @staticmethod
    def _sample_sequence(dataset: LeRobotMixtureDataset, count: int = 32):
        return [
            tuple(int(value) for value in dataset.sample_step(index)[1:])
            for index in range(count)
        ]

    def test_sampling_seed_is_reproducible_but_configurable(self):
        """同 seed 必须复现，而修改 seed 应切换固定的 episode/frame 序列。"""
        first = self._sampling_dataset(seed=42)
        replay = self._sampling_dataset(seed=42)
        different_seed = self._sampling_dataset(seed=43)

        sequence = self._sample_sequence(first)
        self.assertEqual(sequence, self._sample_sequence(replay))
        self.assertNotEqual(sequence, self._sample_sequence(different_seed))


if __name__ == "__main__":
    unittest.main()
