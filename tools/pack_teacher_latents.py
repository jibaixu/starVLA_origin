#!/usr/bin/env python3
"""Pack episode teacher safetensors into one exact, mmap-friendly tensor.

Example:
  python tools/pack_teacher_latents.py \
    --dataset-root /ssdfs/.../libero_goal_no_noops_1.0.0_lerobot \
    --input-relative-dir lingbot_latent_step4_numlayer24_stride16 \
    --output-relative-dir lingbot_latent_step4_numlayer24_stride16_packed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open


def pack(
    dataset_root: Path, input_relative_dir: str, output_relative_dir: str, force: bool = False
) -> Path:
    source = dataset_root / input_relative_dir
    output = dataset_root / output_relative_dir
    paths = []
    for path in source.glob("**/episode_*.safetensors"):
        try:
            episode_id = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        paths.append((episode_id, path))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No episode safetensors found under {source}")
    if len({episode_id for episode_id, _ in paths}) != len(paths):
        raise ValueError("Duplicate episode ids in teacher directory")

    output.mkdir(parents=True, exist_ok=True)
    final_features = output / "features.bin"
    manifest_path = output / "manifest.json"
    if not force and (final_features.exists() or manifest_path.exists()):
        raise FileExistsError(f"Packed output already exists under {output}; use --force to replace it")
    required_bytes = sum(path.stat().st_size for _, path in paths)
    free_bytes = shutil.disk_usage(output).free
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free space for packed teacher: need about {required_bytes} bytes, "
            f"have {free_bytes} bytes"
        )
    temp_features = output / "features.bin.tmp"
    episodes = {}
    shape_tail = None
    dtype_name = None
    offset = 0
    with temp_features.open("wb") as destination:
        for episode_id, path in paths:
            with safe_open(path, framework="pt", device="cpu") as reader:
                metadata = {str(key): str(value) for key, value in (reader.metadata() or {}).items()}
                features = reader.get_tensor("features").contiguous()
                anchors = reader.get_tensor("anchor_frames").to(torch.long)
                if features.ndim != 3 or anchors.ndim != 1 or features.shape[0] != anchors.numel():
                    raise ValueError(f"Invalid feature/anchor shape in {path}")
                current_tail = tuple(features.shape[1:])
                current_dtype = str(features.dtype).replace("torch.", "")
                if shape_tail is None:
                    shape_tail, dtype_name = current_tail, current_dtype
                if current_tail != shape_tail or current_dtype != dtype_name:
                    raise ValueError(f"Inconsistent packed shape/dtype in {path}")
                # bfloat16 has no portable numpy dtype; writing its uint16 view
                # preserves every bit and torch.from_file restores the dtype.
                if features.dtype == torch.bfloat16:
                    features.view(torch.uint16).numpy().tofile(destination)
                else:
                    features.numpy().tofile(destination)
                episodes[str(episode_id)] = {
                    "offset": offset,
                    "count": int(features.shape[0]),
                    "anchors": [int(value) for value in anchors.tolist()],
                    "metadata": metadata,
                }
                offset += int(features.shape[0])
    temp_features.replace(final_features)
    manifest = {
        "format": 1,
        "dtype": dtype_name,
        "feature_shape": [offset, *shape_tail],
        "episodes": episodes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input-relative-dir", required=True)
    parser.add_argument("--output-relative-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(pack(args.dataset_root, args.input_relative_dir, args.output_relative_dir, force=args.force))


if __name__ == "__main__":
    main()
