#!/usr/bin/env python3
"""Plot one aggregate confidence distribution across teacher directories.

Example:
python tools/plot_confidence_distribution.py \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/lingbot_latent_step4_numlayer24_stride16_samples4_confv1 \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/lingbot_latent_step4_numlayer24_stride16_samples4_confv1 \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/lingbot_latent_step4_numlayer24_stride16_samples4_confv1 \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/lingbot_latent_step4_numlayer24_stride16_samples4_confv1 \
    
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1 \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1 \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1 \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1 \
    
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/moto_plan_latent_pre_vq_delta4x4_h16_stride1_epoch10_samples4_confv1 \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/moto_plan_latent_pre_vq_delta4x4_h16_stride1_epoch10_samples4_confv1 \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/moto_plan_latent_pre_vq_delta4x4_h16_stride1_epoch10_samples4_confv1 \
    playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/moto_plan_latent_pre_vq_delta4x4_h16_stride1_epoch10_samples4_confv1 \
    --bins 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open


def _safetensor_paths(input_dirs: Iterable[Path]) -> list[Path]:
    """Return recursively discovered safetensors files, de-duplicated by path."""
    paths: set[Path] = set()
    for directory in input_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"Input confidence directory does not exist: {directory}")
        paths.update(path.resolve() for path in directory.rglob("*.safetensors"))
    ordered = sorted(paths)
    if not ordered:
        raise FileNotFoundError("No .safetensors files found in the input directories")
    return ordered


def _load_confidences(paths: Iterable[Path]) -> np.ndarray:
    values: list[np.ndarray] = []
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            if "confidence" not in reader.keys():
                raise KeyError(f"Missing 'confidence' tensor in {path}")
            confidence = reader.get_tensor("confidence")
        if confidence.ndim != 1:
            raise ValueError(f"Confidence must be one-dimensional in {path}, got {tuple(confidence.shape)}")
        if not torch.is_floating_point(confidence):
            raise ValueError(f"Confidence must be floating point in {path}, got {confidence.dtype}")
        confidence = confidence.float()
        if not torch.isfinite(confidence).all():
            raise ValueError(f"Confidence contains NaN or Inf in {path}")
        if torch.any(confidence < 0.0) or torch.any(confidence > 1.0):
            raise ValueError(f"Confidence must be within [0, 1] in {path}")
        values.append(confidence.numpy())
    merged = np.concatenate(values, axis=0)
    if merged.size == 0:
        raise ValueError("All confidence tensors are empty")
    return merged


def summarize_confidence(values: np.ndarray, bins: int) -> dict:
    """Build JSON-serializable aggregate statistics and fixed-range histogram."""
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    quantile_values = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "num_anchors": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(quantile_values[2]),
        "quantiles": {
            "p05": float(quantile_values[0]),
            "p25": float(quantile_values[1]),
            "p50": float(quantile_values[2]),
            "p75": float(quantile_values[3]),
            "p95": float(quantile_values[4]),
        },
        "zero_ratio": float(np.mean(values == 0.0)),
        "one_ratio": float(np.mean(values == 1.0)),
        "histogram": {
            "bins": int(bins),
            "range": [0.0, 1.0],
            "counts": [int(value) for value in counts.tolist()],
            "edges": [float(value) for value in edges.tolist()],
        },
    }


def plot_confidence(values: np.ndarray, summary: dict, output_path: Path) -> None:
    """Write a fixed [0, 1] aggregate histogram with summary markers."""
    bins = int(summary["histogram"]["bins"])
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.hist(values, bins=bins, range=(0.0, 1.0), density=False, color="#3973ac", alpha=0.82)
    quantiles = summary["quantiles"]
    markers = (
        (summary["mean"], "mean", "#d95f02"),
        (quantiles["p50"], "median", "#1b9e77"),
        (quantiles["p05"], "p05", "#7570b3"),
        (quantiles["p95"], "p95", "#7570b3"),
    )
    for value, label, color in markers:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.5, label=f"{label}={value:.4f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Aggregate teacher confidence distribution (n={summary['num_anchors']:,})"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate confidence tensors from all safetensors under multiple directories."
    )
    parser.add_argument("input_dirs", nargs="+", type=Path, help="Teacher directories to aggregate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("confidence_distribution"),
        help="Directory for confidence_distribution.png/json",
    )
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins (default: 50)")
    args = parser.parse_args()
    if args.bins <= 0:
        parser.error("--bins must be a positive integer")

    paths = _safetensor_paths(args.input_dirs)
    values = _load_confidences(paths)
    summary = summarize_confidence(values, args.bins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "confidence_distribution.png"
    json_path = args.output_dir / "confidence_distribution.json"
    plot_confidence(values, summary, plot_path)
    payload = {
        "input_directories": [str(path.resolve()) for path in args.input_dirs],
        "num_files": len(paths),
        **summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Aggregated {summary['num_anchors']:,} confidence values from {len(paths):,} files")
    print(
        "mean={mean:.6f} median={median:.6f} p05={p05:.6f} p95={p95:.6f} "
        "zero_ratio={zero:.4%}".format(
            mean=summary["mean"],
            median=summary["median"],
            p05=summary["quantiles"]["p05"],
            p95=summary["quantiles"]["p95"],
            zero=summary["zero_ratio"],
        )
    )
    print(f"Wrote {plot_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
