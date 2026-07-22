"""Space2Act 教师、对齐映射、checkpoint 与采样 seed 的核心单元测试。"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from starVLA.dataloader.gr00t_lerobot.teacher_latents import (
    EpisodeTeacherStore,
    TeacherChannelSpec,
)
from starVLA.model.modules.latent_reasoning import CrossAttentionLatentAligner
from starVLA.model.framework.VLM4A.QwenSpace2ActGR00T import QwenSpace2Act_GR00T
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset


def _write_teacher(root: Path, features: torch.Tensor, anchors: torch.Tensor) -> None:
    """按真实目录和字段约定生成一个最小 safetensors 教师文件。"""
    path = root / "latents" / "chunk-000" / "episode_000000.safetensors"
    path.parent.mkdir(parents=True)
    save_file({"features": features, "anchor_frames": anchors}, path)


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
