import json
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

from tools.compress_teacher_latents import fit_projection, load_projection, transform_tree
from tools.build_robotwin_teacher_manifest import build
from tools.build_robotwin_teacher_manifest import _select_group_episodes


def _write_episode(root: Path, name: str, features: torch.Tensor) -> None:
    path = root / name / "chunk-000" / "episode_000000.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "features": features.to(torch.bfloat16),
            "anchor_frames": torch.arange(features.shape[0], dtype=torch.int64),
            "confidence": torch.ones(features.shape[0], dtype=torch.float32),
        },
        str(path),
        metadata={"complete": "true", "share_stride": "8"},
    )


def test_tokenwise_pca_fit_transform_preserves_tokens_and_metadata(tmp_path):
    torch.manual_seed(7)
    source = tmp_path / "source"
    _write_episode(source, "calibration", torch.randn(6, 4, 6))
    _write_episode(source, "other", torch.randn(2, 4, 6))

    mean, components = fit_projection(
        source,
        "features",
        output_dim=3,
        sample_rows=20,
        seed=11,
        directory_name="calibration",
    )
    projection_path = tmp_path / "projection.safetensors"
    from tools.compress_teacher_latents import save_projection

    digest = save_projection(projection_path, mean, components, {"purpose": "test"})
    loaded_mean, loaded_components, metadata = load_projection(projection_path)
    assert torch.equal(mean, loaded_mean)
    assert torch.equal(components, loaded_components)
    assert metadata["projection_sha256"] == digest

    output = tmp_path / "output"
    result = transform_tree(
        source / "calibration",
        output,
        projection_path,
    )
    assert result["episodes"] == 1
    output_path = output / "chunk-000" / "episode_000000.safetensors"
    with safe_open(output_path, framework="pt", device="cpu") as reader:
        assert reader.get_slice("features").get_shape() == [6, 4, 3]
        assert reader.get_slice("features").get_dtype() == "BF16"
        assert torch.equal(reader.get_tensor("anchor_frames"), torch.arange(6))
        assert torch.equal(reader.get_tensor("confidence"), torch.ones(6))
        metadata = reader.metadata()
    assert metadata["feature_transform"] == "tokenwise_pca_v1"
    assert metadata["projection_sha256"] == digest
    assert metadata["source_feature_dim"] == "6"
    assert metadata["feature_dim"] == "3"
    assert json.loads(metadata["source_feature_shape"]) == [6, 4, 6]


def test_projection_fit_directory_filter_rejects_unselected_shape(tmp_path):
    source = tmp_path / "source"
    _write_episode(source, "calibration", torch.randn(4, 3, 5))
    _write_episode(source, "other", torch.randn(4, 3, 7))
    mean, components = fit_projection(
        source,
        "features",
        output_dim=2,
        sample_rows=12,
        directory_name="calibration",
    )
    assert mean.shape == (5,)
    assert components.shape == (2, 5)


def test_episode_balanced_sampling_is_not_dominated_by_long_episode(tmp_path):
    torch.manual_seed(3)
    source = tmp_path / "source"
    _write_episode(source, "calibration/chunk-000/long", torch.randn(20, 3, 6))
    _write_episode(source, "calibration/chunk-000/short", torch.randn(2, 3, 6))
    mean, components = fit_projection(
        source,
        "features",
        output_dim=3,
        sample_rows=6,
        seed=7,
        directory_name="calibration",
        sampling_mode="episode-balanced",
        anchors_per_episode=1,
    )
    assert mean.shape == (6,)
    assert components.shape == (3, 6)


def test_large_feature_dimension_uses_stable_partial_eigh(tmp_path):
    """The cluster MKL build fails in torch.linalg.eigh at 3072 dimensions."""
    torch.manual_seed(17)
    source = tmp_path / "source"
    # 1024 rows satisfy the PCA sample-count guard while keeping the fixture
    # compact; the eigensolver still receives the production-sized 3072-D
    # covariance matrix.
    _write_episode(source, "calibration", torch.randn(1024, 1, 3072))

    mean, components = fit_projection(
        source,
        "features",
        output_dim=1024,
        sample_rows=1024,
        directory_name="calibration",
    )

    assert mean.shape == (3072,)
    assert components.shape == (1024, 3072)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(components).all()


def test_manifest_group_selection_is_deterministic_and_requires_quota():
    episodes = [{"episode_index": index} for index in range(12)]
    first = _select_group_episodes(
        episodes, domain="clean", task_id="task", episodes_per_group=8, seed=42
    )
    second = _select_group_episodes(
        episodes, domain="clean", task_id="task", episodes_per_group=8, seed=42
    )
    assert first == second
    assert len(first) == 8
    try:
        _select_group_episodes(
            episodes[:7], domain="clean", task_id="task", episodes_per_group=8, seed=42
        )
    except ValueError as error:
        assert "cannot select 8" in str(error)
    else:
        raise AssertionError("expected a quota error for an undersized group")


def test_robotwin_teacher_contract_config_matches_compressed_caches():
    config_path = Path(__file__).parents[1] / "examples/simBenchmarks/Robotwin/train_files/space2act_robotwin.yaml"
    config = OmegaConf.load(config_path)
    channels = config.datasets.vla_data.teacher_latents.channels
    assert config.framework.action_model.action_horizon == 50
    assert channels.spatial.anchor_stride == 16
    assert channels.spatial.expected_metadata.cdb_anchor_contract == "shared_future_start"
    assert channels.future.anchor_stride == 16
    assert channels.future.expected_metadata.effective_future_steps == "96"
    assert channels.plan.anchor_stride == 16
    assert channels.plan.target_tokens == 48
    assert channels.plan.expected_metadata.segment_delta == "16"
    assert channels.plan.expected_metadata.segment_count == "6"
    assert channels.plan.expected_metadata.future_horizon_frames == "96"
    assert channels.spatial.expected_metadata.feature_transform == "tokenwise_pca_v1"
    assert channels.future.target_dim == 1024
    assert channels.future.expected_metadata.cdb_anchor_contract == "shared_future_start"
    assert channels.plan.expected_metadata.cdb_anchor_contract == "shared_future_start"


def test_robotwin_manifest_prefers_cam_high(tmp_path):
    (tmp_path / "Randomized").mkdir()
    task = tmp_path / "Clean" / "pick_object"
    (task / "meta").mkdir(parents=True)
    (task / "videos" / "chunk-000").mkdir(parents=True)
    (task / "videos" / "chunk-000" / "observation.images.cam_high" ).touch()
    (task / "videos" / "chunk-000" / "observation.images.image" ).touch()
    (task / "meta" / "info.json").write_text(json.dumps({
        "fps": 15,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}",
        "features": {
            "observation.images.cam_high": {"dtype": "video"},
            "observation.images.image": {"dtype": "video"},
        },
    }))
    (task / "meta" / "episodes.jsonl").write_text(json.dumps({
        "episode_index": 0, "length": 4, "tasks": ["pick object"]
    }) + "\n")
    rows = build(tmp_path)
    assert rows[0]["video_key"] == "observation.images.cam_high"
    assert rows[0]["video_path"].endswith("observation.images.cam_high")


class TeacherCompressionTest(unittest.TestCase):
    def test_tokenwise_pca_fit_transform_preserves_tokens_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            test_tokenwise_pca_fit_transform_preserves_tokens_and_metadata(Path(directory))

    def test_projection_fit_directory_filter_rejects_unselected_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            test_projection_fit_directory_filter_rejects_unselected_shape(Path(directory))

    def test_episode_balanced_sampling_is_not_dominated_by_long_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            test_episode_balanced_sampling_is_not_dominated_by_long_episode(Path(directory))

    def test_manifest_group_selection_is_deterministic_and_requires_quota(self):
        test_manifest_group_selection_is_deterministic_and_requires_quota()

    def test_robotwin_teacher_contract_config_matches_compressed_caches(self):
        test_robotwin_teacher_contract_config_matches_compressed_caches()

    def test_robotwin_manifest_prefers_cam_high(self):
        with tempfile.TemporaryDirectory() as directory:
            test_robotwin_manifest_prefers_cam_high(Path(directory))
