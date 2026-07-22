"""按 episode 懒加载并校验 Space2Act 的空间、未来与规划教师特征。"""

from __future__ import annotations

from dataclasses import dataclass
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
    expected_metadata: dict[str, str] | None = None

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
            expected_metadata={
                str(key): str(value) for key, value in config.get("expected_metadata", {}).items()
            },
        )


class EpisodeTeacherStore:
    """启动时索引 episode 文件，训练时只读取当前样本对应的一个 anchor 切片。"""

    def __init__(self, dataset_path: Path, spec: TeacherChannelSpec, episode_ids) -> None:
        self.dataset_path = Path(dataset_path)
        self.spec = spec
        teacher_root = self.dataset_path / spec.relative_dir
        if not teacher_root.is_dir():
            raise FileNotFoundError(
                f"Space2Act teacher directory for '{spec.name}' does not exist: {teacher_root}"
            )

        self._episode_paths: dict[int, Path] = {}
        self._anchors: dict[int, torch.Tensor] = {}
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
            anchor_values = reader.get_tensor(self.spec.anchor_key).to(torch.long)
            # anchor 很小，可以常驻 CPU；严格递增保证 searchsorted 的时序映射有效。
            if anchor_values.numel() == 0 or torch.any(anchor_values[1:] <= anchor_values[:-1]):
                raise ValueError(f"Teacher anchors must be non-empty and strictly increasing in {path}")
            self._anchors[episode_id] = anchor_values

    def get(self, episode_id: int, frame_index: int) -> torch.Tensor:
        """返回离当前帧最近且不晚于当前帧的 teacher feature。"""
        path = self._episode_paths[int(episode_id)]
        with safe_open(path, framework="pt", device="cpu") as reader:
            anchors = self._anchors[int(episode_id)]
            # right=True 后减一得到 latest anchor <= frame_index；边界 clamp 也使
            # 第一段共享区间和 episode 尾部都能稳定映射到已有 anchor。
            anchor_index = int(torch.searchsorted(anchors, int(frame_index), right=True).item()) - 1
            anchor_index = max(0, min(anchor_index, anchors.numel() - 1))
            return reader.get_slice(self.spec.feature_key)[anchor_index].contiguous()


def build_teacher_stores(dataset_path: Path, data_cfg: Any, episode_ids) -> dict[str, EpisodeTeacherStore]:
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
        stores[str(name)] = EpisodeTeacherStore(dataset_path, spec, episode_ids)
    return stores
