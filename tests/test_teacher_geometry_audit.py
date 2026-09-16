"""Unit tests for the read-only Space2Act teacher geometry audit."""

from __future__ import annotations

import unittest
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from tools.analyze_space2act_teacher_geometry import (
    CHANNEL_SPECS,
    AuditProgress,
    EpisodeWork,
    _confidence_bin_indices,
    _plan_codebook_metrics,
    _read_and_validate_header,
    _run_work,
    _weight_summary,
    channel_specs_for_dataset,
    future_metrics,
    plan_metrics,
    process_episode,
    resolve_progress_mode,
    spatial_metrics,
)


class TeacherGeometryAuditTest(unittest.TestCase):
    def _work(self, episode_id: int, length: int = 3) -> EpisodeWork:
        return EpisodeWork(
            suite="test_suite",
            path=f"/tmp/episode_{episode_id:06d}.safetensors",
            episode_id=episode_id,
            episode_length=length,
            spec=CHANNEL_SPECS["plan"],
            expected_teacher_samples=4,
        )

    def test_confidence_bins_include_one(self):
        values = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertEqual(_confidence_bin_indices(values).tolist(), [0, 1, 2, 3, 4, 4])

    def test_effective_sample_size_handles_zero_confidence(self):
        summary = _weight_summary(np.asarray([0.0, 0.0, 1.0]), minimum_weight=0.0)
        self.assertEqual(summary["zero_weight_ratio"], 2.0 / 3.0)
        self.assertEqual(summary["effective_sample_size"], 1.0)
        softened = _weight_summary(np.asarray([0.0, 0.0, 1.0]), minimum_weight=0.1)
        self.assertEqual(softened["zero_weight_ratio"], 0.0)
        self.assertGreater(softened["effective_sample_size"], 1.0)

    def test_progress_auto_selects_terminal_and_batch_modes(self):
        self.assertEqual(resolve_progress_mode("auto", is_tty=True), "rich")
        self.assertEqual(resolve_progress_mode("auto", is_tty=False), "log")
        self.assertEqual(resolve_progress_mode("off", is_tty=True), "off")

    def test_log_progress_reports_episodes_anchors_and_eta(self):
        clock = [0.0]
        stream = StringIO()
        progress = AuditProgress(
            [self._work(0, length=3), self._work(1, length=5)],
            workers=2,
            requested_mode="log",
            interval_seconds=10.0,
            stream=stream,
            now=lambda: clock[0],
        )
        progress.start()
        clock[0] = 5.0
        progress.complete(self._work(0, length=3), actual_anchors=3)
        clock[0] = 10.0
        progress.complete(self._work(1, length=5), actual_anchors=5)
        progress.stage("summarizing")
        progress.finish(success=True)

        output = stream.getvalue()
        self.assertIn("PROGRESS start", output)
        self.assertIn("PROGRESS running", output)
        self.assertIn("PROGRESS stage=summarizing", output)
        self.assertIn("PROGRESS finished", output)
        self.assertIn("episodes=2/2", output)
        self.assertIn("anchors=8/8", output)
        self.assertIn("channels=plan:2/2", output)
        self.assertIn("eta=0:00:00", output)

    def test_worker_failure_identifies_cache_path_in_log_and_exception(self):
        work = self._work(17)
        stream = StringIO()
        progress = AuditProgress(
            [work],
            workers=1,
            requested_mode="log",
            interval_seconds=1.0,
            stream=stream,
        )
        progress.start()
        with self.assertRaisesRegex(RuntimeError, r"episode 17: /tmp/episode_000017.safetensors"):
            _run_work([work], workers=1, dry_run=True, codebook=None, progress=progress)
        progress.finish(success=False)
        self.assertIn("ERROR failed cache: channel=plan suite=test_suite episode=17", stream.getvalue())

    def test_single_worker_progress_preserves_discovery_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = []
            for episode_id in (0, 1):
                metadata = {
                    "complete": "true",
                    "confidence_aware": "true",
                    "confidence_version": "v1",
                    "num_teacher_samples": "4",
                    "target_aggregation": "mean",
                    "uncertainty_metric": "pairwise_cosine_disagreement",
                    "uncertainty_aggregation": "median_token",
                    "calibration": "train_quantile_05_95",
                    "feature_normalization": "layernorm_last_dim_l2",
                    "seed_policy": "explicit_per_episode_sample",
                    "latent_type": "pre_vq",
                    "segment_offsets": "0:4,4:8,8:12,12:16",
                    "token_order": "segment_major_query_major",
                    "episode_index": str(episode_id),
                    "episode_length": "1",
                    "target_tokens": "32",
                    "feature_dim": "32",
                    "q_low": "0.0",
                    "q_high": "0.2",
                }
                path = root / f"episode_{episode_id:06d}.safetensors"
                save_file(
                    {
                        "features": torch.randn(1, 32, 32, dtype=torch.bfloat16),
                        "anchor_frames": torch.tensor([0], dtype=torch.int64),
                        "valid_future_frames": torch.tensor([0], dtype=torch.int16),
                        "token_ids": torch.zeros((1, 32), dtype=torch.int16),
                        "confidence": torch.ones(1, dtype=torch.float32),
                        "disagreement": torch.zeros(1, dtype=torch.float32),
                    },
                    path,
                    metadata=metadata,
                )
                records.append(
                    EpisodeWork(
                        suite="test_suite",
                        path=str(path),
                        episode_id=episode_id,
                        episode_length=1,
                        spec=CHANNEL_SPECS["plan"],
                        expected_teacher_samples=4,
                    )
                )
            progress = AuditProgress(records, 1, "off", 1.0)
            output = _run_work(records, 1, False, None, progress)
            self.assertEqual([record["episode_id"] for record in output], [0, 1])

    def test_spatial_metrics_detect_neighbor_coherence(self):
        torch.manual_seed(0)
        basis = torch.randn(4, 8)
        grid = basis.repeat_interleave(2, dim=0).repeat_interleave(2, dim=0)
        metrics, _ = spatial_metrics(grid)
        self.assertGreater(
            metrics["spatial_horizontal_neighbor_cosine"],
            metrics["spatial_far_pair_cosine"],
        )

    def test_future_metrics_accept_native_contract(self):
        torch.manual_seed(0)
        feature = torch.randn(1125, 3072, dtype=torch.bfloat16)
        metrics, unit = future_metrics(feature)
        self.assertEqual(tuple(unit.shape), (1125, 3072))
        self.assertIn("future_temporal_dirichlet_energy", metrics)
        self.assertTrue(np.isfinite(metrics["future_temporal_neighbor_cosine"]))

    def test_robotwin_contract_uses_shared_stride16_and_96_frame_horizon(self):
        specs = channel_specs_for_dataset("robotwin")
        self.assertEqual(specs["spatial"].anchor_stride, 16)
        self.assertEqual(specs["future"].anchor_stride, 16)
        self.assertEqual(specs["plan"].anchor_stride, 16)
        self.assertEqual(specs["future"].target_dim, 1024)
        self.assertEqual(specs["plan"].target_tokens, 48)
        self.assertEqual(specs["plan"].valid_future_horizon, 96)
        self.assertIn(("cdb_anchor_contract", "shared_future_start"), specs["plan"].required_metadata)

    def test_robotwin_plan_metrics_support_six_segments(self):
        torch.manual_seed(4)
        feature = torch.randn(48, 32, dtype=torch.bfloat16)
        token_ids = np.arange(48, dtype=np.int16) % 128
        metrics, unit = plan_metrics(feature, token_ids)
        self.assertEqual(tuple(unit.shape), (48, 32))
        self.assertTrue(np.isfinite(metrics["plan_segment_acceleration_l2"]))

    def test_plan_metrics_distinguish_segment_and_query_axes(self):
        torch.manual_seed(0)
        query_vectors = torch.randn(8, 32)
        feature = torch.stack(
            [query_vectors + 0.01 * step * torch.randn(8, 32) for step in range(4)]
        ).reshape(32, 32)
        token_ids = np.arange(32, dtype=np.int16)
        metrics, unit = plan_metrics(feature, token_ids)
        self.assertEqual(tuple(unit.shape), (32, 32))
        self.assertGreater(
            metrics["plan_same_query_next_segment_cosine"],
            metrics["plan_adjacent_query_cosine"],
        )
        self.assertIn("plan_same_query_segment_code_change_rate", metrics)

    def test_plan_metrics_validate_cached_ids_against_codebook(self):
        torch.manual_seed(1)
        codebook = torch.randn(128, 32)
        token_ids = np.arange(32, dtype=np.int16)
        feature = codebook[token_ids].to(torch.bfloat16)
        metrics, _ = plan_metrics(feature, token_ids, codebook=codebook)
        self.assertAlmostEqual(metrics["plan_quantization_mse"], 0.0, places=5)
        self.assertAlmostEqual(metrics["plan_cached_code_id_match_rate"], 1.0, places=6)
        batched = _plan_codebook_metrics(feature.unsqueeze(0), token_ids[None], codebook)
        self.assertAlmostEqual(float(batched["plan_quantization_mse"][0]), 0.0, places=5)
        self.assertAlmostEqual(float(batched["plan_cached_code_id_match_rate"][0]), 1.0, places=6)

    def test_plan_cache_contract_streams_features_and_rejects_pending_cache(self):
        metadata = {
            "complete": "true",
            "confidence_aware": "true",
            "confidence_version": "v1",
            "num_teacher_samples": "4",
            "target_aggregation": "mean",
            "uncertainty_metric": "pairwise_cosine_disagreement",
            "uncertainty_aggregation": "median_token",
            "calibration": "train_quantile_05_95",
            "feature_normalization": "layernorm_last_dim_l2",
            "seed_policy": "explicit_per_episode_sample",
            "latent_type": "pre_vq",
            "segment_offsets": "0:4,4:8,8:12,12:16",
            "token_order": "segment_major_query_major",
            "episode_index": "0",
            "episode_length": "3",
            "target_tokens": "32",
            "feature_dim": "32",
            "q_low": "0.0",
            "q_high": "0.2",
        }
        tensors = {
            "features": torch.randn(3, 32, 32, dtype=torch.bfloat16),
            "anchor_frames": torch.tensor([0, 1, 2], dtype=torch.int64),
            "valid_future_frames": torch.tensor([2, 1, 0], dtype=torch.int16),
            "token_ids": torch.arange(96, dtype=torch.int16).reshape(3, 32) % 128,
            "confidence": torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
            "disagreement": torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episode_000000.safetensors"
            save_file(tensors, path, metadata=metadata)
            work = EpisodeWork(
                suite="test_suite",
                path=str(path),
                episode_id=0,
                episode_length=3,
                spec=CHANNEL_SPECS["plan"],
                expected_teacher_samples=4,
            )
            record = process_episode(work)
            self.assertEqual(record["anchor_count"], 3)
            self.assertAlmostEqual(
                max(record["metrics"]["confidence_calibration_abs_error"]), 0.0, places=6
            )

            pending = dict(metadata, complete="false")
            save_file(tensors, path, metadata=pending)
            with self.assertRaisesRegex(ValueError, "not finalized"):
                _read_and_validate_header(work)


if __name__ == "__main__":
    unittest.main()
