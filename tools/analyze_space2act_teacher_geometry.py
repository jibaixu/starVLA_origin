#!/usr/bin/env python3
"""Audit native geometry and confidence metadata in Space2Act teacher caches.

The tool consumes finalized confidence-aware safetensors caches only.  It never
loads a teacher model or writes back into a cache.  Results are written as a
small JSON report and a flat CSV summary, so a full-cache cluster run can be
reviewed without transferring any teacher feature tensor.

Example:
    python tools/analyze_space2act_teacher_geometry.py \
        --data-root /data/jibaixu/Datasets/LIBERO/libero_no_noop \
        --output-dir /data/jibaixu/analysis/space2act_geometry \
        --workers 4

Add ``--moto-checkpoint /path/to/saved_epoch_10_step_4690`` to include the
pre-VQ quantization error and codebook-margin diagnostics for the plan channel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TextIO

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


DEFAULT_SUITES = (
    "libero_10_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
)


@dataclass(frozen=True)
class ChannelSpec:
    """Static cache contract and native feature layout for one teacher channel."""

    name: str
    default_relative_dir: str
    target_tokens: int
    target_dim: int
    anchor_stride: int
    required_metadata: tuple[tuple[str, str], ...]
    feature_normalization: str
    requires_valid_future_frames: bool = False
    requires_token_ids: bool = False
    valid_future_horizon: int = 16


CHANNEL_SPECS = {
    "spatial": ChannelSpec(
        name="spatial",
        default_relative_dir="lingbot_latent_step4_numlayer24_stride16_samples4_confv1",
        target_tokens=1521,
        target_dim=5120,
        anchor_stride=16,
        required_metadata=(("layer", "24"), ("pooling", "temporal_mean")),
        feature_normalization="layernorm_last_dim",
    ),
    "future": ChannelSpec(
        name="future",
        default_relative_dir="wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1",
        target_tokens=1125,
        target_dim=3072,
        anchor_stride=8,
        required_metadata=(
            ("block_index", "20"),
            ("saved_denoise_step", "19"),
            ("cfg_branch", "positive"),
            ("token_order", "temporal_height_width"),
        ),
        feature_normalization="layernorm_last_dim",
        requires_valid_future_frames=True,
    ),
    "plan": ChannelSpec(
        name="plan",
        default_relative_dir="moto_plan_latent_pre_vq_delta4x4_h16_stride1_epoch10_samples4_confv1",
        target_tokens=32,
        target_dim=32,
        anchor_stride=1,
        required_metadata=(
            ("latent_type", "pre_vq"),
            ("segment_offsets", "0:4,4:8,8:12,12:16"),
            ("token_order", "segment_major_query_major"),
        ),
        feature_normalization="layernorm_last_dim_l2",
        requires_valid_future_frames=True,
        requires_token_ids=True,
    ),
}


def channel_specs_for_dataset(dataset_format: str) -> dict[str, ChannelSpec]:
    """Return cache contracts for LIBERO or the RoboTwin CDB setup."""
    if dataset_format == "libero":
        return CHANNEL_SPECS
    if dataset_format != "robotwin":
        raise ValueError(f"Unsupported dataset format: {dataset_format}")
    return {
        "spatial": replace(
            CHANNEL_SPECS["spatial"],
            default_relative_dir="lingbot_latent_robotwin_stride16_samples4_confv1_pca1024",
            target_dim=1024,
            required_metadata=(
                ("dataset_format", "robotwin"),
                ("layer", "24"),
                ("feature_transform", "tokenwise_pca_v1"),
                ("cdb_anchor_contract", "shared_future_start"),
            ),
        ),
        "future": replace(
            CHANNEL_SPECS["future"],
            default_relative_dir="wan2_2_future_robotwin_stride16_tstride6_samples4_confv1_pca1024",
            anchor_stride=16,
            target_dim=1024,
            valid_future_horizon=96,
            required_metadata=(
                ("dataset_format", "robotwin"),
                ("temporal_stride", "6"),
                ("effective_future_steps", "96"),
                ("feature_transform", "tokenwise_pca_v1"),
                ("cdb_anchor_contract", "shared_future_start"),
            ),
        ),
        "plan": replace(
            CHANNEL_SPECS["plan"],
            default_relative_dir="moto_plan_robotwin_delta16x6_h96_stride16_samples4_confv1",
            target_tokens=48,
            anchor_stride=16,
            valid_future_horizon=96,
            required_metadata=(
                ("dataset_format", "robotwin"),
                ("segment_delta", "16"),
                ("segment_count", "6"),
                ("segment_offsets", "0:16,16:32,32:48,48:64,64:80,80:96"),
                ("cdb_anchor_contract", "shared_future_start"),
            ),
        ),
    }


@dataclass(frozen=True)
class EpisodeWork:
    """Pickle-safe work item for one episode file."""

    suite: str
    path: str
    episode_id: int
    episode_length: int | None
    spec: ChannelSpec
    expected_teacher_samples: int


@dataclass
class EpisodeHeader:
    """Small tensors and validated metadata retained while feature slices stream."""

    metadata: dict[str, str]
    anchors: np.ndarray
    confidence: np.ndarray
    disagreement: np.ndarray
    token_ids: np.ndarray | None
    q_low: float
    q_high: float
    calibration_error: np.ndarray


_WORKER_CODEBOOK: torch.Tensor | None = None


def _estimated_anchor_count(work: EpisodeWork) -> int:
    """Return the expected anchor count from the validated episode index."""
    if work.episode_length is None:
        return 0
    return (work.episode_length + work.spec.anchor_stride - 1) // work.spec.anchor_stride


def resolve_progress_mode(requested: str, is_tty: bool | None = None) -> str:
    """Choose a terminal UI for interactive use and stable lines for batch logs."""
    if requested not in {"auto", "rich", "log", "off"}:
        raise ValueError(f"Unsupported progress mode: {requested}")
    if requested != "auto":
        return requested
    if is_tty is None:
        is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    return "rich" if is_tty else "log"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


class AuditProgress:
    """Main-process-only progress reporter for interactive and Slurm execution."""

    def __init__(
        self,
        work: Sequence[EpisodeWork],
        workers: int,
        requested_mode: str,
        interval_seconds: float,
        stream: TextIO | None = None,
        is_tty: bool | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.work = list(work)
        self.workers = workers
        self.mode = resolve_progress_mode(requested_mode, is_tty)
        self.interval_seconds = interval_seconds
        self.stream = stream if stream is not None else sys.stderr
        self.now = now
        self.started_at: float | None = None
        self.last_log_at: float | None = None
        self.completed_episodes = 0
        self.completed_anchors = 0
        self.total_episodes = len(self.work)
        self.total_anchors = sum(_estimated_anchor_count(item) for item in self.work)
        self.channel_total = {
            channel: sum(1 for item in self.work if item.spec.name == channel)
            for channel in dict.fromkeys(item.spec.name for item in self.work)
        }
        self.channel_completed = {channel: 0 for channel in self.channel_total}
        self.channel_completed_anchors = {channel: 0 for channel in self.channel_total}
        self.suite_total = {
            suite: sum(1 for item in self.work if item.suite == suite)
            for suite in dict.fromkeys(item.suite for item in self.work)
        }
        self.suite_completed = {suite: 0 for suite in self.suite_total}
        self._rich_progress: Any | None = None
        self._rich_total_task: int | None = None
        self._rich_channel_tasks: dict[str, int] = {}
        self._last_item: EpisodeWork | None = None
        self._failed = False

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _write_log(self, state: str, force: bool = False) -> None:
        if self.mode != "log":
            return
        current = self.now()
        if not force and self.last_log_at is not None and current - self.last_log_at < self.interval_seconds:
            return
        elapsed = 0.0 if self.started_at is None else max(0.0, current - self.started_at)
        rate = 0.0 if elapsed == 0.0 else self.completed_episodes / elapsed
        remaining = self.total_episodes - self.completed_episodes
        eta = None if rate == 0.0 else remaining / rate
        channels = ",".join(
            f"{channel}:{self.channel_completed[channel]}/{total}"
            for channel, total in self.channel_total.items()
        )
        suites = ",".join(
            f"{suite}:{self.suite_completed[suite]}/{total}"
            for suite, total in self.suite_total.items()
        )
        last = "none"
        if self._last_item is not None:
            last = (
                f"{self._last_item.spec.name}/{self._last_item.suite}/"
                f"episode_{self._last_item.episode_id:06d}"
            )
        print(
            f"[{self._timestamp()}] PROGRESS {state}: "
            f"episodes={self.completed_episodes}/{self.total_episodes} "
            f"anchors={self.completed_anchors}/{self.total_anchors} "
            f"rate={rate:.2f} episode/s elapsed={_format_duration(elapsed)} "
            f"eta={_format_duration(eta)} channels={channels} suites={suites} last={last}",
            file=self.stream,
            flush=True,
        )
        self.last_log_at = current

    def _start_rich(self) -> None:
        try:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )
        except ImportError as error:
            raise RuntimeError("--progress rich requires the 'rich' package") from error
        self._rich_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("anchors {task.fields[anchors]}/{task.fields[anchor_total]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=Console(file=self.stream, force_terminal=True),
            transient=False,
            refresh_per_second=4,
        )
        self._rich_progress.start()
        self._rich_total_task = self._rich_progress.add_task(
            "all channels",
            total=self.total_episodes,
            anchors=0,
            anchor_total=self.total_anchors,
        )
        for channel, total in self.channel_total.items():
            anchor_total = sum(
                _estimated_anchor_count(item) for item in self.work if item.spec.name == channel
            )
            self._rich_channel_tasks[channel] = self._rich_progress.add_task(
                channel,
                total=total,
                anchors=0,
                anchor_total=anchor_total,
            )

    def start(self) -> None:
        self.started_at = self.now()
        if self.mode == "rich":
            self._start_rich()
        elif self.mode == "log":
            self._write_log("start", force=True)

    def complete(self, work: EpisodeWork, actual_anchors: int) -> None:
        self._last_item = work
        self.completed_episodes += 1
        self.completed_anchors += actual_anchors
        self.channel_completed[work.spec.name] += 1
        self.suite_completed[work.suite] += 1
        if self.mode == "rich":
            assert self._rich_progress is not None and self._rich_total_task is not None
            self._rich_progress.update(
                self._rich_total_task,
                advance=1,
                anchors=self.completed_anchors,
            )
            channel_task = self._rich_channel_tasks[work.spec.name]
            self.channel_completed_anchors[work.spec.name] += actual_anchors
            self._rich_progress.update(
                channel_task,
                advance=1,
                anchors=self.channel_completed_anchors[work.spec.name],
            )
        else:
            self._write_log("running")

    def stage(self, name: str) -> None:
        if self.mode == "rich":
            assert self._rich_progress is not None
            self._rich_progress.console.print(f"[bold]Stage:[/bold] {name}")
        elif self.mode == "log":
            self._write_log(f"stage={name}", force=True)

    def heartbeat(self) -> None:
        """Emit a batch-log heartbeat while all workers are still busy."""
        self._write_log("running", force=True)

    def failure(self, work: EpisodeWork, error: BaseException) -> None:
        self._failed = True
        self._last_item = work
        if self.mode == "rich" and self._rich_progress is not None:
            self._rich_progress.console.print(
                f"[bold red]Failed:[/bold red] {work.spec.name}/{work.suite}/"
                f"episode_{work.episode_id:06d}: {error}"
            )
        elif self.mode == "log":
            self._write_log("failed", force=True)
            print(
                f"[{self._timestamp()}] ERROR failed cache: channel={work.spec.name} "
                f"suite={work.suite} episode={work.episode_id} path={work.path}: {error}",
                file=self.stream,
                flush=True,
            )

    def finish(self, success: bool) -> None:
        if self.mode == "rich" and self._rich_progress is not None:
            self._rich_progress.stop()
        elif self.mode == "log":
            self._write_log("finished" if success else "aborted", force=True)


def _canonical_dtype(dtype: str) -> str:
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "f16": "float16",
        "float16": "float16",
        "f32": "float32",
        "float32": "float32",
        "i16": "int16",
        "int16": "int16",
        "i32": "int32",
        "int32": "int32",
        "i64": "int64",
        "int64": "int64",
    }
    return aliases.get(str(dtype).lower(), str(dtype).lower())


def _episode_id_from_path(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot parse episode id from {path}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metadata_float(metadata: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            try:
                parsed = float(value)
            except ValueError as error:
                raise ValueError(f"Metadata {key}={value!r} is not a float") from error
            _require(math.isfinite(parsed), f"Metadata {key} must be finite")
            return parsed
    raise ValueError(f"Missing calibration metadata; expected one of {keys}")


def _expected_anchors(length: int, stride: int) -> torch.Tensor:
    return torch.arange(0, length, stride, dtype=torch.int64)


def _expected_valid_future(length: int, anchors: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.clamp(length - 1 - anchors, min=0, max=horizon).to(torch.int16)


def _validate_metadata(
    metadata: dict[str, str], work: EpisodeWork, path: Path
) -> tuple[float, float]:
    spec = work.spec
    _require(metadata.get("complete", "false").lower() == "true", f"Cache is not finalized: {path}")
    _require(
        metadata.get("confidence_aware", "false").lower() == "true",
        f"Cache is not confidence-aware: {path}",
    )
    _require(
        metadata.get("target_aggregation") == "mean",
        f"Cache target aggregation must be mean: {path}",
    )
    _require(
        metadata.get("confidence_version") == "v1",
        f"Unexpected confidence version in {path}",
    )
    _require(
        metadata.get("uncertainty_metric") == "pairwise_cosine_disagreement",
        f"Unexpected uncertainty metric in {path}",
    )
    _require(
        metadata.get("uncertainty_aggregation") == "median_token",
        f"Unexpected uncertainty aggregation in {path}",
    )
    _require(
        metadata.get("calibration") == "train_quantile_05_95",
        f"Cache is not finalized with train quantile calibration: {path}",
    )
    _require(
        metadata.get("feature_normalization") == spec.feature_normalization,
        f"Unexpected feature normalization in {path}",
    )
    _require("seed_policy" in metadata, f"Missing seed policy in {path}")
    _require(
        metadata.get("num_teacher_samples") == str(work.expected_teacher_samples),
        f"Unexpected num_teacher_samples in {path}",
    )
    if "episode_index" in metadata:
        _require(
            int(metadata["episode_index"]) == work.episode_id,
            f"episode_index metadata does not match file name: {path}",
        )
    if work.episode_length is not None and "episode_length" in metadata:
        _require(
            int(metadata["episode_length"]) == work.episode_length,
            f"episode_length metadata does not match dataset index: {path}",
        )
    for key, expected in spec.required_metadata:
        _require(
            metadata.get(key) == expected,
            f"Metadata {key}={metadata.get(key)!r}, expected {expected!r}: {path}",
        )
    if "target_tokens" in metadata:
        _require(
            int(metadata["target_tokens"]) == spec.target_tokens,
            f"target_tokens metadata mismatch: {path}",
        )
    if "feature_dim" in metadata:
        _require(
            int(metadata["feature_dim"]) == spec.target_dim,
            f"feature_dim metadata mismatch: {path}",
        )
    q_low = _metadata_float(metadata, "calibration_q_low", "q_low")
    q_high = _metadata_float(metadata, "calibration_q_high", "q_high")
    _require(q_low <= q_high, f"Invalid calibration interval in {path}")
    return q_low, q_high


def _read_and_validate_header(work: EpisodeWork) -> EpisodeHeader:
    path = Path(work.path)
    spec = work.spec
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        metadata = {str(key): str(value) for key, value in (reader.metadata() or {}).items()}
        q_low, q_high = _validate_metadata(metadata, work, path)
        keys = set(reader.keys())
        required = {"features", "anchor_frames", "confidence", "disagreement"}
        if spec.requires_valid_future_frames:
            required.add("valid_future_frames")
        if spec.requires_token_ids:
            required.add("token_ids")
        _require(required.issubset(keys), f"Missing tensors {sorted(required - keys)} in {path}")

        feature_slice = reader.get_slice("features")
        feature_shape = tuple(feature_slice.get_shape())
        _require(
            len(feature_shape) == 3
            and feature_shape[1:] == (spec.target_tokens, spec.target_dim),
            f"Feature shape mismatch in {path}: got {feature_shape}",
        )
        _require(
            _canonical_dtype(feature_slice.get_dtype()) == "bfloat16",
            f"Features must be BF16 in {path}",
        )

        anchor_slice = reader.get_slice("anchor_frames")
        _require(
            tuple(anchor_slice.get_shape()) == (feature_shape[0],)
            and _canonical_dtype(anchor_slice.get_dtype()) == "int64",
            f"Invalid anchor tensor in {path}",
        )
        anchors = reader.get_tensor("anchor_frames").to(torch.int64)
        _require(
            anchors.numel() > 0 and int(anchors[0]) == 0 and torch.all(anchors[1:] > anchors[:-1]),
            f"Anchors must start at zero and be strictly increasing in {path}",
        )
        if anchors.numel() > 1:
            _require(
                torch.all(torch.diff(anchors) == spec.anchor_stride),
                f"Anchor stride must be {spec.anchor_stride} in {path}",
            )
        if work.episode_length is not None:
            expected = _expected_anchors(work.episode_length, spec.anchor_stride)
            _require(torch.equal(anchors, expected), f"Anchor coverage mismatch in {path}")

        confidence_slice = reader.get_slice("confidence")
        disagreement_slice = reader.get_slice("disagreement")
        _require(
            tuple(confidence_slice.get_shape()) == (feature_shape[0],)
            and _canonical_dtype(confidence_slice.get_dtype()) == "float32",
            f"Invalid confidence tensor in {path}",
        )
        _require(
            tuple(disagreement_slice.get_shape()) == (feature_shape[0],)
            and _canonical_dtype(disagreement_slice.get_dtype()) == "float32",
            f"Invalid disagreement tensor in {path}",
        )
        confidence = reader.get_tensor("confidence").float()
        disagreement = reader.get_tensor("disagreement").float()
        _require(torch.isfinite(confidence).all(), f"Non-finite confidence in {path}")
        _require(torch.isfinite(disagreement).all(), f"Non-finite disagreement in {path}")
        _require(torch.all((0.0 <= confidence) & (confidence <= 1.0)), f"Confidence outside [0, 1] in {path}")
        _require(torch.all(disagreement >= 0.0), f"Negative disagreement in {path}")

        token_ids = None
        if spec.requires_valid_future_frames:
            valid_slice = reader.get_slice("valid_future_frames")
            _require(
                tuple(valid_slice.get_shape()) == (feature_shape[0],)
                and _canonical_dtype(valid_slice.get_dtype()) == "int16",
                f"Invalid valid_future_frames tensor in {path}",
            )
            if work.episode_length is not None:
                valid = reader.get_tensor("valid_future_frames").to(torch.int16)
                _require(
                    torch.equal(valid, _expected_valid_future(work.episode_length, anchors, spec.valid_future_horizon)),
                    f"valid_future_frames mismatch in {path}",
                )
        if spec.requires_token_ids:
            ids_slice = reader.get_slice("token_ids")
            _require(
                tuple(ids_slice.get_shape()) == (feature_shape[0], spec.target_tokens)
                and _canonical_dtype(ids_slice.get_dtype()) == "int16",
                f"Invalid token_ids tensor in {path}",
            )
            token_ids = reader.get_tensor("token_ids").to(torch.int64).numpy()
            _require(
                np.all((0 <= token_ids) & (token_ids < 128)),
                f"Moto token ids must be in [0, 128) in {path}",
            )

    expected_confidence = 1.0 - torch.clamp(
        (disagreement - q_low) / (q_high - q_low + 1e-8), min=0.0, max=1.0
    )
    return EpisodeHeader(
        metadata=metadata,
        anchors=anchors.numpy(),
        confidence=confidence.numpy(),
        disagreement=disagreement.numpy(),
        token_ids=token_ids,
        q_low=q_low,
        q_high=q_high,
        calibration_error=(expected_confidence - confidence).abs().numpy(),
    )


def _normalized_tokens(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return LayerNorm features and their unit-direction counterpart."""
    normalized = F.layer_norm(feature.float(), (feature.shape[-1],))
    return normalized, F.normalize(normalized, dim=-1)


def _mean_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left * right).sum(dim=-1).mean().item())


def _mean_square_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).square().mean().item())


def _base_metrics(feature: torch.Tensor) -> dict[str, float]:
    raw = feature.float()
    token_norms = torch.linalg.vector_norm(raw, dim=-1)
    token_stds = raw.std(dim=-1, unbiased=False)
    return {
        "raw_token_norm_mean": float(token_norms.mean().item()),
        "raw_token_norm_std": float(token_norms.std(unbiased=False).item()),
        "raw_last_dim_std_mean": float(token_stds.mean().item()),
    }


def spatial_metrics(feature: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    """Compute 2-D grid local structure from one LingBot target."""
    token_count, feature_dim = feature.shape
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise ValueError(f"Spatial feature token count must be square, got {token_count}")
    normalized, unit = _normalized_tokens(feature)
    normalized = normalized.reshape(side, side, feature_dim)
    unit = unit.reshape(side, side, feature_dim)
    flat_unit = unit.reshape(-1, feature_dim)
    return {
        **_base_metrics(feature),
        "spatial_horizontal_neighbor_cosine": _mean_cosine(unit[:, :-1], unit[:, 1:]),
        "spatial_vertical_neighbor_cosine": _mean_cosine(unit[:-1], unit[1:]),
        "spatial_far_pair_cosine": _mean_cosine(flat_unit, flat_unit.roll(token_count // 2, dims=0)),
        "spatial_dirichlet_energy": 0.5
        * (
            _mean_square_delta(normalized[:, :-1], normalized[:, 1:])
            + _mean_square_delta(normalized[:-1], normalized[1:])
        ),
    }, unit.reshape(token_count, feature_dim)


def future_metrics(feature: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    """Compute spatial and temporal local structure from one Wan target."""
    if feature.ndim != 2 or feature.shape[0] != 1125:
        raise ValueError(f"Future feature must be [1125,D], got {tuple(feature.shape)}")
    feature_dim = feature.shape[1]
    normalized, unit = _normalized_tokens(feature)
    normalized = normalized.reshape(5, 15, 15, feature_dim)
    unit = unit.reshape(5, 15, 15, feature_dim)
    flat_unit = unit.reshape(1125, feature_dim)
    horizontal_energy = _mean_square_delta(normalized[:, :, :-1], normalized[:, :, 1:])
    vertical_energy = _mean_square_delta(normalized[:, :-1], normalized[:, 1:])
    temporal_energy = _mean_square_delta(normalized[:-1], normalized[1:])
    return {
        **_base_metrics(feature),
        "future_horizontal_neighbor_cosine": _mean_cosine(unit[:, :, :-1], unit[:, :, 1:]),
        "future_vertical_neighbor_cosine": _mean_cosine(unit[:, :-1], unit[:, 1:]),
        "future_temporal_neighbor_cosine": _mean_cosine(unit[:-1], unit[1:]),
        "future_far_pair_cosine": _mean_cosine(flat_unit, flat_unit.roll(1125 // 2, dims=0)),
        "future_spatial_dirichlet_energy": 0.5 * (horizontal_energy + vertical_energy),
        "future_temporal_dirichlet_energy": temporal_energy,
        "future_delta_from_first_temporal_group": _mean_square_delta(normalized[1:], normalized[:1]),
    }, unit.reshape(1125, 3072)


def plan_metrics(
    feature: torch.Tensor,
    token_ids: np.ndarray,
    codebook: torch.Tensor | None = None,
) -> tuple[dict[str, float], torch.Tensor]:
    """Respect Moto's segment-major/query-major layout; queries are not a chain."""
    if feature.ndim != 2 or feature.shape[1] != 32 or feature.shape[0] % 8:
        raise ValueError(f"Plan feature must be [segments*8,32], got {tuple(feature.shape)}")
    if tuple(token_ids.shape) != (feature.shape[0],):
        raise ValueError(f"Plan token ids must match feature rows, got {tuple(token_ids.shape)}")
    segment_count = feature.shape[0] // 8
    normalized, unit = _normalized_tokens(feature)
    raw = feature.float().reshape(segment_count, 8, 32)
    normalized = normalized.reshape(segment_count, 8, 32)
    unit = unit.reshape(segment_count, 8, 32)
    ids = torch.from_numpy(token_ids).reshape(segment_count, 8)
    raw_delta = raw[1:] - raw[:-1]
    metrics = {
        **_base_metrics(feature),
        "plan_same_query_next_segment_cosine": _mean_cosine(unit[:-1], unit[1:]),
        "plan_adjacent_query_cosine": _mean_cosine(unit[:, :-1], unit[:, 1:]),
        "plan_opposite_query_cosine": _mean_cosine(unit, unit.roll(4, dims=1)),
        "plan_segment_delta_l2": float(torch.linalg.vector_norm(raw_delta, dim=-1).mean().item()),
        "plan_segment_acceleration_l2": float(
            torch.linalg.vector_norm(raw[2:] - 2.0 * raw[1:-1] + raw[:-2], dim=-1).mean().item()
        ),
        "plan_same_query_segment_code_change_rate": float((ids[1:] != ids[:-1]).float().mean().item()),
        "plan_adjacent_query_code_change_rate": float((ids[:, 1:] != ids[:, :-1]).float().mean().item()),
        "plan_active_code_count": float(torch.unique(ids).numel()),
    }
    if codebook is not None:
        points = feature.float().reshape(-1, 32)
        squared_distance = torch.cdist(points, codebook.float()).square()
        nearest = squared_distance.topk(k=2, largest=False, dim=-1)
        token_tensor = torch.from_numpy(token_ids).to(torch.long)
        metrics.update(
            {
                "plan_quantization_mse": float((nearest.values[:, 0] / 32.0).mean().item()),
                "plan_codebook_margin": float((nearest.values[:, 1] - nearest.values[:, 0]).mean().item()),
                "plan_cached_code_id_match_rate": float(
                    (nearest.indices[:, 0] == token_tensor).float().mean().item()
                ),
            }
        )
    return metrics, unit.reshape(feature.shape[0], 32)


def _plan_codebook_metrics(
    features: torch.Tensor, token_ids: np.ndarray, codebook: torch.Tensor
) -> dict[str, np.ndarray]:
    """Compute Moto VQ diagnostics for every anchor in one episode at once."""
    _require(features.ndim == 3 and features.shape[2] == 32 and features.shape[1] % 8 == 0, "Invalid plan batch")
    _require(
        token_ids.shape == (features.shape[0], features.shape[1]),
        "Plan token ids do not match feature anchors",
    )
    _require(codebook.ndim == 2 and codebook.shape[1] == 32, "Invalid Moto codebook")
    _require(
        np.all((0 <= token_ids) & (token_ids < codebook.shape[0])),
        "Cached Moto token id exceeds codebook size",
    )
    points = features.float().reshape(-1, 32)
    codebook = codebook.float()
    # Squared L2 distances as one matrix multiplication is much faster than a
    # separate cdist call for each [32, 32] anchor.
    squared_distance = (
        points.square().sum(dim=1, keepdim=True)
        + codebook.square().sum(dim=1).unsqueeze(0)
        - 2.0 * points @ codebook.t()
    ).clamp_min_(0.0)
    nearest = squared_distance.topk(k=2, largest=False, dim=-1)
    shape = (features.shape[0], features.shape[1])
    nearest_distance = nearest.values[:, 0].reshape(shape)
    margin = (nearest.values[:, 1] - nearest.values[:, 0]).reshape(shape)
    nearest_ids = nearest.indices[:, 0].reshape(shape)
    cached_ids = torch.from_numpy(token_ids).to(dtype=torch.long)
    return {
        "plan_quantization_mse": (nearest_distance.mean(dim=1) / 32.0).numpy(),
        "plan_codebook_margin": margin.mean(dim=1).numpy(),
        "plan_cached_code_id_match_rate": (nearest_ids == cached_ids).float().mean(dim=1).numpy(),
    }


def _add_previous_anchor_metrics(
    metrics: dict[str, float],
    current_unit: torch.Tensor,
    previous_unit: torch.Tensor | None,
    prefix: str,
    current_ids: np.ndarray | None = None,
    previous_ids: np.ndarray | None = None,
) -> None:
    if previous_unit is None:
        metrics[f"{prefix}_previous_anchor_same_position_cosine"] = float("nan")
        if current_ids is not None:
            metrics[f"{prefix}_previous_anchor_same_slot_code_change_rate"] = float("nan")
        return
    metrics[f"{prefix}_previous_anchor_same_position_cosine"] = _mean_cosine(
        current_unit, previous_unit
    )
    if current_ids is not None and previous_ids is not None:
        metrics[f"{prefix}_previous_anchor_same_slot_code_change_rate"] = float(
            np.mean(current_ids != previous_ids)
        )


def _init_worker(codebook: np.ndarray | None) -> None:
    global _WORKER_CODEBOOK
    _WORKER_CODEBOOK = torch.from_numpy(codebook) if codebook is not None else None


def process_episode(work: EpisodeWork) -> dict[str, Any]:
    """Validate and stream one episode; only scalar metrics leave the worker."""
    header = _read_and_validate_header(work)
    metrics: dict[str, list[float]] = {}
    previous_unit = None
    previous_ids = None
    with safe_open(work.path, framework="pt", device="cpu") as reader:
        feature_slice = reader.get_slice("features")
        feature_batch = None
        codebook_metrics = None
        if work.spec.name == "plan" and _WORKER_CODEBOOK is not None:
            assert header.token_ids is not None
            feature_batch = reader.get_tensor("features")
            codebook_metrics = _plan_codebook_metrics(
                feature_batch, header.token_ids, _WORKER_CODEBOOK
            )
        for anchor_index in range(header.confidence.size):
            feature = feature_slice[anchor_index] if feature_batch is None else feature_batch[anchor_index]
            if not torch.isfinite(feature).all():
                raise ValueError(
                    f"Non-finite {work.spec.name} feature at anchor index {anchor_index}: {work.path}"
                )
            if work.spec.name == "spatial":
                current_metrics, current_unit = spatial_metrics(feature)
                _add_previous_anchor_metrics(
                    current_metrics, current_unit, previous_unit, "spatial"
                )
            elif work.spec.name == "future":
                current_metrics, current_unit = future_metrics(feature)
                _add_previous_anchor_metrics(
                    current_metrics, current_unit, previous_unit, "future"
                )
            else:
                assert header.token_ids is not None
                current_ids = header.token_ids[anchor_index]
                current_metrics, current_unit = plan_metrics(
                    feature, current_ids
                )
                if codebook_metrics is not None:
                    current_metrics.update(
                        {
                            name: float(values[anchor_index])
                            for name, values in codebook_metrics.items()
                        }
                    )
                _add_previous_anchor_metrics(
                    current_metrics,
                    current_unit,
                    previous_unit,
                    "plan",
                    current_ids=current_ids,
                    previous_ids=previous_ids,
                )
                previous_ids = current_ids
            for name, value in current_metrics.items():
                metrics.setdefault(name, []).append(value)
            previous_unit = current_unit

    metrics["confidence"] = header.confidence.astype(np.float64, copy=False).tolist()
    metrics["raw_disagreement"] = header.disagreement.astype(np.float64, copy=False).tolist()
    metrics["confidence_calibration_abs_error"] = header.calibration_error.astype(
        np.float64, copy=False
    ).tolist()
    return {
        "suite": work.suite,
        "channel": work.spec.name,
        "episode_id": work.episode_id,
        "path": work.path,
        "anchor_count": int(header.confidence.size),
        "q_low": header.q_low,
        "q_high": header.q_high,
        "metadata": {
            key: header.metadata[key]
            for key in (
                "teacher",
                "checkpoint_sha256",
                "num_teacher_samples",
                "target_aggregation",
                "calibration_scope",
                "layer",
                "block_index",
                "saved_denoise_step",
                "latent_type",
            )
            if key in header.metadata
        },
        "metrics": metrics,
    }


def inspect_episode(work: EpisodeWork) -> dict[str, Any]:
    """Header-only validation path used by --dry-run."""
    header = _read_and_validate_header(work)
    return {
        "suite": work.suite,
        "channel": work.spec.name,
        "episode_id": work.episode_id,
        "path": work.path,
        "anchor_count": int(header.confidence.size),
        "q_low": header.q_low,
        "q_high": header.q_high,
        "metadata": {
            key: header.metadata[key]
            for key in ("teacher", "checkpoint_sha256", "calibration_scope")
            if key in header.metadata
        },
    }


def _load_episode_index(suite_root: Path) -> dict[int, int]:
    path = suite_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LeRobot episode index: {path}")
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["episode_index"])] = int(row["length"])
    if not rows:
        raise ValueError(f"Episode index is empty: {path}")
    return rows


def discover_robotwin_suites(data_root: Path) -> list[str]:
    """Return deterministic ``robotwin_<domain>_<task>`` suite names."""
    suites: list[str] = []
    for domain in ("Clean", "Randomized"):
        domain_root = data_root / domain
        if not domain_root.is_dir():
            continue
        for task_root in sorted(domain_root.iterdir()):
            if task_root.is_dir() and (task_root / "meta" / "episodes.jsonl").is_file():
                suites.append(f"robotwin_{domain.lower()}_{task_root.name}")
    if not suites:
        raise FileNotFoundError(f"No RoboTwin task datasets found under {data_root}")
    return suites


def discover_work(
    data_root: Path,
    suites: Sequence[str],
    specs: Sequence[ChannelSpec],
    relative_dirs: dict[str, str],
    expected_teacher_samples: int,
    allow_partial: bool,
    dataset_format: str = "libero",
) -> tuple[list[EpisodeWork], dict[str, dict[str, dict[str, int]]]]:
    work: list[EpisodeWork] = []
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for suite in suites:
        if dataset_format == "robotwin" and suite.startswith("robotwin_"):
            _, domain, task = suite.split("_", 2)
            suite_root = data_root / domain.capitalize() / task
        else:
            suite_root = data_root / suite
        episode_lengths = _load_episode_index(suite_root)
        coverage[suite] = {}
        for spec in specs:
            teacher_root = suite_root / relative_dirs[spec.name]
            if not teacher_root.is_dir():
                raise FileNotFoundError(f"Missing {spec.name} cache directory: {teacher_root}")
            paths = sorted(teacher_root.glob("**/episode_*.safetensors"))
            by_episode: dict[int, Path] = {}
            for path in paths:
                episode_id = _episode_id_from_path(path)
                if episode_id in by_episode:
                    raise ValueError(f"Duplicate {spec.name} cache for episode {episode_id}: {path}")
                by_episode[episode_id] = path
            unknown = sorted(set(by_episode) - set(episode_lengths))
            if unknown:
                raise ValueError(f"{spec.name} cache contains unknown episodes in {suite}: {unknown[:10]}")
            missing = sorted(set(episode_lengths) - set(by_episode))
            if missing and not allow_partial:
                raise FileNotFoundError(
                    f"{spec.name} cache under {teacher_root} is incomplete; missing episodes {missing[:10]}"
                )
            selected_ids = sorted(by_episode)
            if not selected_ids:
                raise FileNotFoundError(f"No {spec.name} cache files found under {teacher_root}")
            coverage[suite][spec.name] = {
                "dataset_episodes": len(episode_lengths),
                "cached_episodes": len(selected_ids),
                "missing_episodes": len(missing),
            }
            work.extend(
                EpisodeWork(
                    suite=suite,
                    path=str(by_episode[episode_id]),
                    episode_id=episode_id,
                    episode_length=episode_lengths[episode_id],
                    spec=spec,
                    expected_teacher_samples=expected_teacher_samples,
                )
                for episode_id in selected_ids
            )
    return work, coverage


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _summary_statistics(values: np.ndarray) -> dict[str, float | int | None]:
    values = _finite(values)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "minimum": None,
            "maximum": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }
    quantiles = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
    }


def _episode_block_ci(
    per_episode_values: list[np.ndarray], samples: int, seed: int
) -> tuple[float | None, float | None]:
    valid = [_finite(values) for values in per_episode_values]
    valid = [values for values in valid if values.size]
    if len(valid) < 2 or samples <= 0:
        return None, None
    sums = np.asarray([values.sum(dtype=np.float64) for values in valid])
    counts = np.asarray([values.size for values in valid], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(valid), size=(samples, len(valid)))
    bootstrap = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(low), float(high)


def _metric_summary(
    records: Sequence[dict[str, Any]], metric: str, bootstrap_samples: int, seed: int, bin_index: int | None
) -> dict[str, float | int | None]:
    values_by_episode = []
    for record in records:
        values = np.asarray(record["metrics"][metric], dtype=np.float64)
        if bin_index is not None:
            confidence = np.asarray(record["metrics"]["confidence"], dtype=np.float64)
            values = values[_confidence_bin_indices(confidence) == bin_index]
        values_by_episode.append(values)
    merged = np.concatenate(values_by_episode) if values_by_episode else np.empty(0, dtype=np.float64)
    summary = _summary_statistics(merged)
    low, high = _episode_block_ci(values_by_episode, bootstrap_samples, seed)
    summary["bootstrap_ci_low"] = low
    summary["bootstrap_ci_high"] = high
    return summary


CONFIDENCE_EDGES = np.linspace(0.0, 1.0, num=6)
CONFIDENCE_BIN_LABELS = (
    "[0.0,0.2)",
    "[0.2,0.4)",
    "[0.4,0.6)",
    "[0.6,0.8)",
    "[0.8,1.0]",
)


def _confidence_bin_indices(confidence: np.ndarray) -> np.ndarray:
    # Keep exact decimal boundaries in their upper bin despite binary-float noise
    # from np.linspace (for example, 0.6 versus 0.6000000000000001).
    return np.minimum(np.floor(confidence * 5.0 + 1e-12).astype(np.int64), 4)


def _weight_summary(confidence: np.ndarray, minimum_weight: float) -> dict[str, float | int]:
    weights = minimum_weight + (1.0 - minimum_weight) * confidence
    sum_weights = float(weights.sum(dtype=np.float64))
    sum_squares = float(np.square(weights).sum(dtype=np.float64))
    effective_sample_size = 0.0 if sum_squares == 0.0 else sum_weights**2 / sum_squares
    return {
        "minimum_weight": minimum_weight,
        "sum_weight": sum_weights,
        "mean_weight": float(weights.mean()),
        "zero_weight_ratio": float(np.mean(weights == 0.0)),
        "effective_sample_size": effective_sample_size,
        "effective_sample_ratio": effective_sample_size / float(weights.size),
    }


def _metadata_audit(records: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {}
    for record in records:
        for key, value in record["metadata"].items():
            values.setdefault(key, set()).add(str(value))
    return {key: sorted(value) for key, value in sorted(values.items())}


def summarize_records(
    records: Sequence[dict[str, Any]], bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty set of episode records")
    metrics = sorted({name for record in records for name in record["metrics"]})
    expected_metrics = set(records[0]["metrics"])
    if any(set(record["metrics"]) != expected_metrics for record in records):
        raise ValueError("Metric set differs across records; cache contracts are inconsistent")
    confidence = np.concatenate(
        [np.asarray(record["metrics"]["confidence"], dtype=np.float64) for record in records]
    )
    q_lows = np.asarray([record["q_low"] for record in records], dtype=np.float64)
    q_highs = np.asarray([record["q_high"] for record in records], dtype=np.float64)
    calibration = {
        "q_low_values": sorted({float(value) for value in q_lows.tolist()}),
        "q_high_values": sorted({float(value) for value in q_highs.tolist()}),
        "q_low_consistent": bool(np.allclose(q_lows, q_lows[0], rtol=0.0, atol=1e-12)),
        "q_high_consistent": bool(np.allclose(q_highs, q_highs[0], rtol=0.0, atol=1e-12)),
    }
    report = {
        "episode_count": len(records),
        "anchor_count": int(confidence.size),
        "metadata_audit": _metadata_audit(records),
        "confidence_weights": [_weight_summary(confidence, value) for value in (0.0, 0.1, 0.2)],
        "calibration": calibration,
        "metrics": {},
        "confidence_bins": {},
    }
    for metric in metrics:
        metric_seed = seed + zlib.crc32(metric.encode("utf-8"))
        report["metrics"][metric] = _metric_summary(
            records, metric, bootstrap_samples, metric_seed, bin_index=None
        )
    for bin_index, label in enumerate(CONFIDENCE_BIN_LABELS):
        selected = _confidence_bin_indices(confidence) == bin_index
        bin_report = {
            "anchor_count": int(selected.sum()),
            "confidence_range": [float(CONFIDENCE_EDGES[bin_index]), float(CONFIDENCE_EDGES[bin_index + 1])],
            "metrics": {},
        }
        for metric in metrics:
            metric_seed = seed + zlib.crc32(f"{label}:{metric}".encode("utf-8"))
            bin_report["metrics"][metric] = _metric_summary(
                records, metric, bootstrap_samples, metric_seed, bin_index=bin_index
            )
        report["confidence_bins"][label] = bin_report
    return report


def _load_moto_codebook(path: Path) -> np.ndarray:
    weights_path = path / "pytorch_model.bin" if path.is_dir() else path
    if not weights_path.is_file():
        raise FileNotFoundError(f"Moto checkpoint file does not exist: {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Moto checkpoint is not a state dict: {weights_path}")
    candidates = [
        value
        for key, value in checkpoint.items()
        if key.endswith("vector_quantizer.embedding.weight") and isinstance(value, torch.Tensor)
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one vector_quantizer.embedding.weight in Moto checkpoint, "
            f"found {len(candidates)}"
        )
    codebook = candidates[0].detach().float().cpu()
    if codebook.ndim != 2 or codebook.shape[1] != 32 or codebook.shape[0] < 2:
        raise ValueError(f"Unexpected Moto codebook shape: {tuple(codebook.shape)}")
    return codebook.numpy()


def _run_work(
    work: Sequence[EpisodeWork],
    workers: int,
    dry_run: bool,
    codebook: np.ndarray | None,
    progress: AuditProgress,
) -> list[dict[str, Any]]:
    function = inspect_episode if dry_run else process_episode
    if workers <= 1:
        _init_worker(codebook)
        records = []
        for item in work:
            try:
                record = function(item)
            except Exception as error:
                progress.failure(item, error)
                raise RuntimeError(
                    f"Failed to process {item.spec.name} cache for {item.suite} "
                    f"episode {item.episode_id}: {item.path}"
                ) from error
            progress.complete(item, int(record["anchor_count"]))
            records.append(record)
        return records
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(codebook,),
    ) as executor:
        futures = {
            executor.submit(function, item): index
            for index, item in enumerate(work)
        }
        ordered_records: list[dict[str, Any] | None] = [None] * len(work)
        pending = set(futures)
        timeout = progress.interval_seconds if progress.mode == "log" else None
        while pending:
            completed, pending = wait(
                pending,
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not completed:
                progress.heartbeat()
                continue
            for future in completed:
                index = futures[future]
                item = work[index]
                try:
                    record = future.result()
                except Exception as error:
                    for outstanding in pending:
                        outstanding.cancel()
                    progress.failure(item, error)
                    raise RuntimeError(
                        f"Failed to process {item.spec.name} cache for {item.suite} "
                        f"episode {item.episode_id}: {item.path}"
                    ) from error
                ordered_records[index] = record
                progress.complete(item, int(record["anchor_count"]))
    if any(record is None for record in ordered_records):
        raise RuntimeError("A worker completed without producing an audit record")
    return [record for record in ordered_records if record is not None]


def _csv_rows(report: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for channel, channel_report in report["channels"].items():
        for scope, scope_report in channel_report["scopes"].items():
            for metric, summary in scope_report.get("metrics", {}).items():
                yield {
                    "channel": channel,
                    "scope": scope,
                    "confidence_bin": "all",
                    "metric": metric,
                    **summary,
                }
            for label, bin_report in scope_report.get("confidence_bins", {}).items():
                for metric, summary in bin_report["metrics"].items():
                    yield {
                        "channel": channel,
                        "scope": scope,
                        "confidence_bin": label,
                        "metric": metric,
                        **summary,
                    }


def _write_report(output_dir: Path, report: dict[str, Any], overwrite: bool) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "teacher_geometry_report.json"
    csv_path = output_dir / "teacher_geometry_summary.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist: {', '.join(str(path) for path in existing)}; use --overwrite"
        )
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(_csv_rows(report))
    fieldnames = [
        "channel",
        "scope",
        "confidence_bin",
        "metric",
        "count",
        "mean",
        "std",
        "minimum",
        "maximum",
        "p05",
        "p50",
        "p95",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _console_summary(report: dict[str, Any]) -> None:
    print("Space2Act teacher geometry audit")
    for channel, channel_report in report["channels"].items():
        for scope, scope_report in channel_report["scopes"].items():
            if scope == "all_suites":
                continue
            confidence = scope_report["metrics"]["confidence"]
            disagreement = scope_report["metrics"]["raw_disagreement"]
            print(
                f"  {channel:7s} {scope:45s} anchors={scope_report['anchor_count']:7d} "
                f"confidence p50={confidence['p50']:.5f} "
                f"disagreement p50={disagreement['p50']:.5f}"
            )
            structure_metric = next(
                (
                    name
                    for name in scope_report["metrics"]
                    if name.endswith("neighbor_cosine") or name.endswith("next_segment_cosine")
                ),
                None,
            )
            if structure_metric is not None:
                structure = scope_report["metrics"][structure_metric]
                ci_low = structure["bootstrap_ci_low"]
                ci_high = structure["bootstrap_ci_high"]
                ci_text = "unavailable" if ci_low is None or ci_high is None else f"[{ci_low:.5f}, {ci_high:.5f}]"
                print(f"    {structure_metric}={structure['mean']:.5f} 95% episode-bootstrap CI={ci_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only, full-cache geometry and confidence audit for Space2Act teachers."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing LIBERO suite roots")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON and CSV reports")
    parser.add_argument("--dataset-format", choices=("libero", "robotwin"), default="libero", help="Select LIBERO or RoboTwin cache contracts")
    parser.add_argument("--suites", nargs="*", default=None, help="Suite names (auto-discovered for RoboTwin)")
    parser.add_argument(
        "--channels", nargs="*", choices=sorted(CHANNEL_SPECS), default=sorted(CHANNEL_SPECS)
    )
    parser.add_argument("--spatial-dir", default=None, help="Override the dataset-specific Spatial cache directory")
    parser.add_argument("--future-dir", default=None, help="Override the dataset-specific Future cache directory")
    parser.add_argument("--plan-dir", default=None, help="Override the dataset-specific Plan cache directory")
    parser.add_argument("--workers", type=int, default=1, help="CPU reader workers; default: 1")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-teacher-samples", type=int, default=4)
    parser.add_argument("--moto-checkpoint", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true", help="Analyze available episode files only")
    parser.add_argument("--dry-run", action="store_true", help="Validate cache headers without reading features")
    parser.add_argument(
        "--progress",
        choices=("auto", "rich", "log", "off"),
        default="auto",
        help="Progress display: auto uses Rich on a TTY and stable log lines otherwise",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between non-interactive progress log lines; default: 30",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the two report files if they exist")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    if args.expected_teacher_samples <= 0:
        parser.error("--expected-teacher-samples must be positive")
    if args.progress_interval <= 0.0:
        parser.error("--progress-interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    channel_specs = channel_specs_for_dataset(args.dataset_format)
    suites = args.suites or (
        discover_robotwin_suites(data_root) if args.dataset_format == "robotwin" else list(DEFAULT_SUITES)
    )
    relative_dirs = {
        "spatial": args.spatial_dir or channel_specs["spatial"].default_relative_dir,
        "future": args.future_dir or channel_specs["future"].default_relative_dir,
        "plan": args.plan_dir or channel_specs["plan"].default_relative_dir,
    }
    specs = [channel_specs[name] for name in args.channels]
    work, coverage = discover_work(
        data_root,
        suites,
        specs,
        relative_dirs,
        args.expected_teacher_samples,
        args.allow_partial,
        args.dataset_format,
    )
    progress = AuditProgress(
        work,
        workers=args.workers,
        requested_mode=args.progress,
        interval_seconds=args.progress_interval,
    )
    if progress.mode != "off":
        print(
            "Space2Act teacher geometry audit: "
            f"episodes={len(work):,} estimated_anchors={progress.total_anchors:,} "
            f"workers={args.workers} mode={progress.mode} "
            f"channels={','.join(args.channels)} suites={','.join(suites)} "
            f"moto_codebook={'enabled' if args.moto_checkpoint else 'disabled'}",
            file=sys.stderr,
            flush=True,
        )
    codebook = None
    if args.moto_checkpoint is not None:
        if "plan" not in args.channels:
            raise ValueError("--moto-checkpoint requires --channels to include plan")
        codebook = _load_moto_codebook(args.moto_checkpoint)
    progress.start()
    try:
        records = _run_work(work, args.workers, args.dry_run, codebook, progress)
        progress.stage("summarizing")
    except BaseException:
        progress.finish(success=False)
        raise
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "header_audit" if args.dry_run else "full_geometry",
        "inputs": {
            "data_root": str(data_root),
            "suites": list(suites),
            "channels": list(args.channels),
            "relative_directories": relative_dirs,
            "workers": args.workers,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "expected_teacher_samples": args.expected_teacher_samples,
            "moto_checkpoint": str(args.moto_checkpoint) if args.moto_checkpoint else None,
            "progress": args.progress,
            "progress_mode": progress.mode,
            "progress_interval": args.progress_interval,
        },
        "coverage": coverage,
        "channels": {},
    }
    try:
        for channel in args.channels:
            channel_records = [record for record in records if record["channel"] == channel]
            scopes: dict[str, Any] = {}
            if args.dry_run:
                for suite in suites:
                    suite_records = [record for record in channel_records if record["suite"] == suite]
                    scopes[suite] = {
                        "episode_count": len(suite_records),
                        "anchor_count": int(sum(record["anchor_count"] for record in suite_records)),
                        "q_low_values": sorted({record["q_low"] for record in suite_records}),
                        "q_high_values": sorted({record["q_high"] for record in suite_records}),
                        "metadata_audit": _metadata_audit(suite_records),
                    }
            else:
                for suite in suites:
                    suite_records = [record for record in channel_records if record["suite"] == suite]
                    scopes[suite] = summarize_records(
                        suite_records, args.bootstrap_samples, args.seed
                    )
                scopes["all_suites"] = summarize_records(
                    channel_records, args.bootstrap_samples, args.seed
                )
            report["channels"][channel] = {"scopes": scopes}
        progress.stage("writing reports")
        json_path, csv_path = _write_report(args.output_dir, report, args.overwrite)
    except BaseException:
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    if args.dry_run:
        print(f"Validated {len(records):,} episode cache files without reading feature tensors")
    else:
        _console_summary(report)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
