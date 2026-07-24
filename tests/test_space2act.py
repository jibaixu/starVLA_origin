"""Space2Act 教师、对齐映射、checkpoint 与采样 seed 的核心单元测试。"""

from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from starVLA.dataloader.gr00t_lerobot.teacher_latents import (
    EpisodeTeacherStore,
    TeacherChannelSpec,
)
from starVLA.model.modules.latent_reasoning import CrossAttentionLatentAligner
from starVLA.model.framework.VLM4A.QwenSpace2ActGR00T import QwenSpace2Act_GR00T
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

    def test_stage2_yaml_and_shell_contract(self):
        """Stage 2 配置应固定 teacher shape，并保持 shell 顶部直接赋默认值。"""
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
