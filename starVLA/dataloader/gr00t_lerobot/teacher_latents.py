"""按 episode 懒加载并校验 Space2Act 的空间、未来与规划教师特征。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class TeacherChannelSpec:
    """单个教师通道的磁盘位置、目标 shape 与 safetensors key 契约。"""

    name: str
    relative_dir: str
    target_tokens: int
    target_dim: int
    feature_key: str = "features"
    anchor_key: str = "anchor_frames"
    feature_dtype: str | None = None
    anchor_dtype: str | None = None
    anchor_stride: int | None = None
    validate_episode_length: bool = False
    expected_metadata: dict[str, str] | None = None
    packed_relative_dir: str | None = None

    @classmethod
    def from_config(cls, name: str, config: Any) -> "TeacherChannelSpec":
        """将 OmegaConf/dict 配置规范化为不可变规格，并尽早拒绝缺失字段。"""
        relative_dir = config.get("relative_dir", None)
        target_tokens = config.get("target_tokens", None)
        target_dim = config.get("target_dim", None)
        missing = [
            key
            for key, value in (
                ("relative_dir", relative_dir),
                ("target_tokens", target_tokens),
                ("target_dim", target_dim),
            )
            if value in (None, "", "null")
        ]
        if missing:
            raise ValueError(
                f"Space2Act teacher channel '{name}' is enabled but is missing: {', '.join(missing)}"
            )
        return cls(
            name=name,
            relative_dir=str(relative_dir),
            target_tokens=int(target_tokens),
            target_dim=int(target_dim),
            feature_key=str(config.get("feature_key", "features")),
            anchor_key=str(config.get("anchor_key", "anchor_frames")),
            feature_dtype=(
                str(config.get("feature_dtype")) if config.get("feature_dtype", None) is not None else None
            ),
            anchor_dtype=(
                str(config.get("anchor_dtype")) if config.get("anchor_dtype", None) is not None else None
            ),
            anchor_stride=(
                int(config.get("anchor_stride")) if config.get("anchor_stride", None) is not None else None
            ),
            validate_episode_length=config.get("validate_episode_length", False)
            in (True, "True", "true", 1),
            expected_metadata={
                str(key): str(value) for key, value in config.get("expected_metadata", {}).items()
            },
            packed_relative_dir=(
                str(config.get("packed_relative_dir"))
                if config.get("packed_relative_dir", None) not in (None, "", "null")
                else None
            ),
        )


def _canonical_dtype(dtype: str) -> str:
    """Normalize safetensors header names and config names to one spelling."""
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
    normalized = str(dtype).lower()
    return aliases.get(normalized, normalized)


class EpisodeTeacherStore:
    """启动时索引 episode 文件，训练时只读取当前样本对应的一个 anchor 切片。"""

    def __init__(
        self,
        dataset_path: Path,
        spec: TeacherChannelSpec,
        episode_ids,
        episode_lengths: dict[int, int] | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.spec = spec
        self._episode_lengths = {
            int(episode_id): int(length) for episode_id, length in (episode_lengths or {}).items()
        }
        teacher_root = self.dataset_path / spec.relative_dir

        self._episode_paths: dict[int, Path] = {}
        self._anchors: dict[int, torch.Tensor] = {}
        self._packed_features: torch.Tensor | None = None
        self._packed_offsets: dict[int, int] = {}
        self._packed_counts: dict[int, int] = {}
        packed_root = self.dataset_path / spec.packed_relative_dir if spec.packed_relative_dir else None
        if packed_root is not None:
            if not packed_root.is_dir():
                raise FileNotFoundError(f"Packed teacher directory does not exist: {packed_root}")
            self._load_packed(packed_root, episode_ids, episode_lengths)
            return
        if not teacher_root.is_dir():
            raise FileNotFoundError(
                f"Space2Act teacher directory for '{spec.name}' does not exist: {teacher_root}"
            )
        # 文件名中的 episode id 是数据集内的主键。提前检查重复文件，避免
        # glob 顺序决定训练时实际使用哪一份 teacher，造成不可复现实验。
        for path in teacher_root.glob("**/episode_*.safetensors"):
            try:
                episode_id = int(path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if episode_id in self._episode_paths:
                raise ValueError(f"Duplicate teacher file for episode {episode_id}: {path}")
            self._episode_paths[episode_id] = path

        # 只要求当前 dataset view 中的 episode 齐全；episode_indices 调试子集
        # 因而不需要提前提取整个数据集的 teacher。
        requested = [int(episode_id) for episode_id in episode_ids]
        missing = [episode_id for episode_id in requested if episode_id not in self._episode_paths]
        if missing:
            preview = ", ".join(map(str, missing[:10]))
            raise FileNotFoundError(
                f"Missing '{spec.name}' teacher files under {teacher_root} for episodes: {preview}"
            )
        for episode_id in requested:
            self._validate_episode(episode_id)

    def _validate_episode(self, episode_id: int) -> None:
        """验证 metadata、tensor key/shape 与 anchor 单调性，并缓存小型 anchor 向量。"""
        path = self._episode_paths[episode_id]
        with safe_open(path, framework="pt", device="cpu") as reader:
            metadata = reader.metadata() or {}
            # complete=false 通常表示提取进程中断，不能把部分写入的文件用于监督。
            if metadata.get("complete", "true").lower() != "true":
                raise ValueError(f"Teacher file is marked incomplete: {path}")
            # expected_metadata 用于锁定层号、stride 和 feature 维度，防止目录名正确
            # 但实际由另一套提取参数生成的文件被静默加载。
            for key, expected_value in (self.spec.expected_metadata or {}).items():
                actual_value = metadata.get(key)
                if actual_value != expected_value:
                    raise ValueError(
                        f"Teacher metadata mismatch in {path}: {key}={actual_value!r}, "
                        f"expected {expected_value!r}"
                    )
            if "episode_index" in metadata and int(metadata["episode_index"]) != episode_id:
                raise ValueError(
                    f"Teacher metadata episode_index={metadata['episode_index']} does not match {path}"
                )
            if "suite" in metadata:
                suite = metadata["suite"]
                dataset_name = self.dataset_path.name
                if dataset_name != suite and not dataset_name.startswith(f"{suite}_"):
                    raise ValueError(
                        f"Teacher metadata suite={suite!r} does not match dataset {dataset_name}"
                    )
            keys = set(reader.keys())
            required = {self.spec.feature_key, self.spec.anchor_key}
            if not required.issubset(keys):
                raise ValueError(f"Teacher file {path} is missing tensors {sorted(required - keys)}")
            features = reader.get_slice(self.spec.feature_key)
            anchors = reader.get_slice(self.spec.anchor_key)
            # 只读取 slice header 来检查大 tensor 的 shape；此处不把 features
            # 整体搬入内存。约定为 [anchor数, teacher token数, teacher维度]。
            feature_shape = tuple(features.get_shape())
            anchor_shape = tuple(anchors.get_shape())
            feature_dtype = _canonical_dtype(features.get_dtype())
            anchor_dtype = _canonical_dtype(anchors.get_dtype())
            expected_tail = (self.spec.target_tokens, self.spec.target_dim)
            if len(feature_shape) != 3 or feature_shape[1:] != expected_tail:
                raise ValueError(
                    f"Teacher '{self.spec.name}' shape mismatch in {path}: "
                    f"expected [A,{expected_tail[0]},{expected_tail[1]}], got {feature_shape}"
                )
            if len(anchor_shape) != 1 or anchor_shape[0] != feature_shape[0]:
                raise ValueError(
                    f"Teacher anchors mismatch in {path}: features={feature_shape}, anchors={anchor_shape}"
                )
            if self.spec.feature_dtype is not None and feature_dtype != _canonical_dtype(
                self.spec.feature_dtype
            ):
                raise ValueError(
                    f"Teacher feature dtype mismatch in {path}: got {feature_dtype}, "
                    f"expected {_canonical_dtype(self.spec.feature_dtype)}"
                )
            if self.spec.anchor_dtype is not None and anchor_dtype != _canonical_dtype(
                self.spec.anchor_dtype
            ):
                raise ValueError(
                    f"Teacher anchor dtype mismatch in {path}: got {anchor_dtype}, "
                    f"expected {_canonical_dtype(self.spec.anchor_dtype)}"
                )
            anchor_values = reader.get_tensor(self.spec.anchor_key).to(torch.long)
            # anchor 很小，可以常驻 CPU；严格递增保证 searchsorted 的时序映射有效。
            if anchor_values.numel() == 0 or torch.any(anchor_values[1:] <= anchor_values[:-1]):
                raise ValueError(f"Teacher anchors must be non-empty and strictly increasing in {path}")
            if int(anchor_values[0]) != 0:
                raise ValueError(f"Teacher anchors must start at frame 0 in {path}")
            if self.spec.anchor_stride is not None and anchor_values.numel() > 1:
                strides = anchor_values[1:] - anchor_values[:-1]
                if torch.any(strides != self.spec.anchor_stride):
                    raise ValueError(
                        f"Teacher anchor stride mismatch in {path}: expected {self.spec.anchor_stride}"
                    )

            episode_length = self._episode_lengths.get(episode_id)
            if self.spec.validate_episode_length:
                if episode_length is None:
                    raise ValueError(f"Episode length is required to validate teacher anchors in {path}")
                metadata_length = metadata.get("episode_length")
                if metadata_length is None or int(metadata_length) != episode_length:
                    raise ValueError(
                        f"Teacher episode_length mismatch in {path}: metadata={metadata_length}, "
                        f"dataset={episode_length}"
                    )
                if self.spec.anchor_stride is None:
                    raise ValueError("validate_episode_length requires anchor_stride")
                expected_anchors = torch.arange(0, episode_length, self.spec.anchor_stride)
                if not torch.equal(anchor_values, expected_anchors):
                    raise ValueError(
                        f"Teacher anchors do not cover episode {episode_id} with stride "
                        f"{self.spec.anchor_stride} in {path}"
                    )
            self._anchors[episode_id] = anchor_values

    def _load_packed(
        self, packed_root: Path, episode_ids, episode_lengths: dict[int, int] | None
    ) -> None:
        """Load an optional exact packed representation without opening episode files per sample."""
        manifest_path = packed_root / "manifest.json"
        features_path = packed_root / "features.bin"
        if not manifest_path.is_file() or not features_path.is_file():
            raise FileNotFoundError(f"Packed teacher store is incomplete under {packed_root}")
        manifest = json.loads(manifest_path.read_text())
        shape = tuple(manifest.get("feature_shape", ()))
        expected_tail = (self.spec.target_tokens, self.spec.target_dim)
        if len(shape) != 3 or tuple(shape[1:]) != expected_tail:
            raise ValueError(
                f"Packed teacher '{self.spec.name}' shape mismatch: expected [A,{expected_tail[0]},{expected_tail[1]}], got {shape}"
            )
        dtype_name = _canonical_dtype(manifest.get("dtype", ""))
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported packed teacher dtype: {manifest.get('dtype')!r}")
        episodes = manifest.get("episodes", {})
        requested = [int(episode_id) for episode_id in episode_ids]
        missing = [episode_id for episode_id in requested if str(episode_id) not in episodes]
        if missing:
            raise FileNotFoundError(f"Packed teacher '{self.spec.name}' is missing episodes: {missing[:10]}")
        total_elements = int(shape[0]) * int(shape[1]) * int(shape[2])
        expected_bytes = total_elements * torch.empty((), dtype=dtype).element_size()
        if features_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Packed teacher file size mismatch: got {features_path.stat().st_size}, expected {expected_bytes}"
            )
        self._packed_features = torch.from_file(
            str(features_path), shared=False, size=total_elements, dtype=dtype
        ).view(shape)
        for episode_id in requested:
            info = episodes[str(episode_id)]
            metadata = {str(key): str(value) for key, value in info.get("metadata", {}).items()}
            for key, expected_value in (self.spec.expected_metadata or {}).items():
                if metadata.get(key) != expected_value:
                    raise ValueError(
                        f"Packed teacher metadata mismatch for episode {episode_id}: "
                        f"{key}={metadata.get(key)!r}, expected {expected_value!r}"
                    )
            if "episode_index" in metadata and int(metadata["episode_index"]) != episode_id:
                raise ValueError(f"Packed teacher episode_index mismatch for episode {episode_id}")
            anchors = torch.tensor(info["anchors"], dtype=torch.long)
            if anchors.numel() == 0 or torch.any(anchors[1:] <= anchors[:-1]):
                raise ValueError(f"Packed teacher anchors are not strictly increasing for episode {episode_id}")
            self._anchors[episode_id] = anchors
            self._packed_offsets[episode_id] = int(info["offset"])
            self._packed_counts[episode_id] = int(info["count"])
            if self._packed_counts[episode_id] != anchors.numel():
                raise ValueError(f"Packed teacher count mismatch for episode {episode_id}")
            if self._packed_offsets[episode_id] < 0 or (
                self._packed_offsets[episode_id] + self._packed_counts[episode_id] > shape[0]
            ):
                raise ValueError(f"Packed teacher offset is out of bounds for episode {episode_id}")
            if self.spec.anchor_stride is not None and anchors.numel() > 1:
                if torch.any(anchors[1:] - anchors[:-1] != self.spec.anchor_stride):
                    raise ValueError(f"Packed teacher anchor stride mismatch for episode {episode_id}")
            if self.spec.validate_episode_length:
                length = (episode_lengths or {}).get(episode_id)
                if length is None:
                    raise ValueError("Episode length is required to validate packed teacher anchors")
                expected = torch.arange(0, int(length), int(self.spec.anchor_stride))
                if not torch.equal(anchors, expected):
                    raise ValueError(f"Packed teacher anchors do not cover episode {episode_id}")

    def get(self, episode_id: int, frame_index: int) -> torch.Tensor:
        """返回离当前帧最近且不晚于当前帧的 teacher feature。"""
        if self._packed_features is not None:
            episode_id = int(episode_id)
            anchors = self._anchors[episode_id]
            anchor_index = int(torch.searchsorted(anchors, int(frame_index), right=True).item()) - 1
            anchor_index = max(0, min(anchor_index, anchors.numel() - 1))
            row = self._packed_offsets[episode_id] + anchor_index
            # A row view from torch.from_file is already contiguous, so
            # ``contiguous()`` would keep the entire mmap-backed storage.
            # Clone the row before DataLoader IPC; otherwise a worker tries to
            # share the complete packed file for every sample and can SIGBUS.
            return self._packed_features[row].clone()
        path = self._episode_paths[int(episode_id)]
        with safe_open(path, framework="pt", device="cpu") as reader:
            anchors = self._anchors[int(episode_id)]
            # right=True 后减一得到 latest anchor <= frame_index；边界 clamp 也使
            # 第一段共享区间和 episode 尾部都能稳定映射到已有 anchor。
            anchor_index = int(torch.searchsorted(anchors, int(frame_index), right=True).item()) - 1
            anchor_index = max(0, min(anchor_index, anchors.numel() - 1))
            return reader.get_slice(self.spec.feature_key)[anchor_index].contiguous()


def build_teacher_stores(
    dataset_path: Path,
    data_cfg: Any,
    episode_ids,
    episode_lengths: dict[int, int] | None = None,
) -> dict[str, EpisodeTeacherStore]:
    """仅为 enabled 的 S/F/P 通道创建 store，未启用通道不触发文件依赖。"""
    teacher_cfg = data_cfg.get("teacher_latents", None) if data_cfg is not None else None
    if not teacher_cfg:
        return {}
    channels = teacher_cfg.get("channels", {})
    stores = {}
    for name, channel_cfg in channels.items():
        if channel_cfg.get("enabled", False) not in (True, "True", "true", 1):
            continue
        spec = TeacherChannelSpec.from_config(str(name), channel_cfg)
        stores[str(name)] = EpisodeTeacherStore(
            dataset_path, spec, episode_ids, episode_lengths=episode_lengths
        )
    return stores
