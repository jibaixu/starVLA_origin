#!/usr/bin/env python3
"""Fit and apply a token-wise PCA projection to teacher latent caches.

The projection is shared by every token position, but tokens are never mixed:
an input cache with shape ``[A, T, D]`` becomes ``[A, T, K]`` by applying the
same centered ``D -> K`` projection to each ``[D]`` vector.  Episode files,
anchors, confidence values, and auxiliary tensors are copied to a sibling
output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _top_eigenvectors(covariance: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Return the largest covariance eigenvectors in descending order.

    PyTorch's MKL-backed ``SSYEVD`` fails for the 3072/5120-dimensional
    covariance matrices used by the teacher features on some cluster nodes.
    SciPy's ``evr`` driver avoids that backend path and computes only the
    requested leading eigenvectors.
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Expected a square covariance matrix, got {tuple(covariance.shape)}")
    if output_dim <= 0 or output_dim > covariance.shape[0]:
        raise ValueError(
            f"output_dim={output_dim} must be in [1, {covariance.shape[0]}]"
        )
    if not torch.isfinite(covariance).all():
        finite = covariance[torch.isfinite(covariance)]
        value_range = "empty" if finite.numel() == 0 else (
            f"[{finite.min().item():.6g}, {finite.max().item():.6g}]"
        )
        raise ValueError(
            "Covariance contains NaN or Inf values; "
            f"shape={tuple(covariance.shape)}, finite_range={value_range}"
        )

    try:
        from scipy.linalg import eigh as scipy_eigh
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for PCA fitting of large teacher feature dimensions. "
            "Install the project's requirements before running compression."
        ) from error

    input_dim = covariance.shape[0]
    # SciPy returns eigenvalues/eigenvectors in ascending order.
    _, eigenvectors = scipy_eigh(
        covariance.detach().cpu().numpy(),
        subset_by_index=(input_dim - output_dim, input_dim - 1),
        driver="evr",
        check_finite=False,
        overwrite_a=False,
    )
    components = torch.from_numpy(eigenvectors[:, ::-1].copy()).T.contiguous().float()
    if not torch.isfinite(components).all():
        raise ValueError(
            "SciPy returned non-finite PCA components; "
            f"shape={tuple(components.shape)}"
        )
    return components


def episode_paths(root: Path, directory_name: str | None = None) -> list[Path]:
    paths = sorted(root.glob("**/episode_*.safetensors"))
    if directory_name:
        paths = [path for path in paths if directory_name in path.parts]
    if not paths:
        raise FileNotFoundError(f"No episode safetensors found under {root}")
    return paths


def _feature_header(path: Path, feature_key: str) -> tuple[tuple[int, ...], str]:
    with safe_open(path, framework="pt", device="cpu") as reader:
        view = reader.get_slice(feature_key)
        shape = tuple(view.get_shape())
        if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] <= 0:
            raise ValueError(f"Expected [anchors,tokens,dim] in {path}, got {shape}")
        return shape, view.get_dtype()


def fit_projection(
    input_root: Path,
    feature_key: str,
    output_dim: int,
    sample_rows: int = 100_000,
    seed: int = 42,
    directory_name: str | None = None,
    sampling_mode: str = "global",
    anchors_per_episode: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a centered PCA projection and return ``(mean, components)``.

    ``components`` has shape ``[output_dim, input_dim]`` and is ordered by
    decreasing eigenvalue.  Sampling is deterministic and bounded by
    ``sample_rows`` flattened token vectors.
    """
    if output_dim <= 0 or sample_rows <= 0:
        raise ValueError("output_dim and sample_rows must be positive")
    if sampling_mode not in {"global", "episode-balanced"}:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
    if anchors_per_episode <= 0:
        raise ValueError("anchors_per_episode must be positive")
    paths = episode_paths(input_root, directory_name)
    first_shape, first_dtype = _feature_header(paths[0], feature_key)
    input_dim = first_shape[2]
    if output_dim > input_dim:
        raise ValueError(f"output_dim={output_dim} exceeds input_dim={input_dim}")
    headers = [(path, *_feature_header(path, feature_key)) for path in paths]
    for path, shape, dtype in headers[1:]:
        if shape[1] != first_shape[1] or shape[2] != input_dim or dtype != first_dtype:
            raise ValueError(f"Inconsistent feature shape/dtype in {path}: {shape}, {dtype}")

    tokens = first_shape[1]
    rng = random.Random(seed)
    selected_by_file: dict[Path, list[int]] = {}
    token_indices_by_anchor: dict[tuple[Path, int], torch.Tensor] = {}
    if sampling_mode == "global":
        total_anchors = sum(shape[0] for _, shape, _ in headers)
        anchor_quota = min(total_anchors, max(1, (int(sample_rows) + tokens - 1) // tokens))
        selected_global = sorted(rng.sample(range(total_anchors), anchor_quota))
        cursor = 0
        global_offset = 0
        for path, shape, _ in headers:
            local = []
            while cursor < len(selected_global) and selected_global[cursor] < global_offset + shape[0]:
                local.append(selected_global[cursor] - global_offset)
                cursor += 1
            selected_by_file[path] = local
            global_offset += shape[0]
        token_quota = tokens
    else:
        for path, shape, _ in headers:
            quota = min(anchors_per_episode, shape[0])
            selected = sorted(rng.sample(range(shape[0]), quota))
            selected_by_file[path] = selected
        selected_anchor_count = sum(len(indices) for indices in selected_by_file.values())
        if selected_anchor_count <= 0 or sample_rows < selected_anchor_count:
            raise ValueError(
                "sample_rows is too small for episode-balanced sampling: "
                f"need at least {selected_anchor_count} token rows, got {sample_rows}"
            )
        token_quota = min(tokens, max(1, int(sample_rows) // selected_anchor_count))
        for path, indices in selected_by_file.items():
            for anchor in indices:
                if token_quota == tokens:
                    token_indices = torch.arange(tokens, dtype=torch.long)
                else:
                    token_indices = torch.tensor(
                        sorted(rng.sample(range(tokens), token_quota)), dtype=torch.long
                    )
                token_indices_by_anchor[(path, anchor)] = token_indices

    mean_sum = torch.zeros(input_dim, dtype=torch.float64)
    second_sum = torch.zeros((input_dim, input_dim), dtype=torch.float64)
    seen = 0
    for path, shape, _ in headers:
        indices = selected_by_file[path]
        with safe_open(path, framework="pt", device="cpu") as reader:
            feature_view = reader.get_slice(feature_key)
            for anchor in indices:
                rows = feature_view[anchor]
                if sampling_mode == "episode-balanced":
                    rows = rows[token_indices_by_anchor[(path, anchor)]]
                rows = rows.to(dtype=torch.float64).reshape(-1, input_dim)
                mean_sum += rows.sum(dim=0)
                second_sum += rows.T @ rows
                seen += rows.shape[0]
    if seen < max(2, output_dim):
        raise ValueError(f"Not enough sampled rows for PCA: {seen}")
    mean = mean_sum / float(seen)
    covariance = (second_sum / float(seen) - torch.outer(mean, mean)).float()
    covariance = (covariance + covariance.T) * 0.5
    components = _top_eigenvectors(covariance, output_dim)
    return mean.float().contiguous(), components


def projection_sha256(mean: torch.Tensor, components: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (mean.detach().cpu().float().contiguous(), components.detach().cpu().float().contiguous()):
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_projection(path: Path, mean: torch.Tensor, components: torch.Tensor, metadata: dict[str, str]) -> str:
    digest = projection_sha256(mean, components)
    payload = {str(key): str(value) for key, value in metadata.items()}
    payload.update({
        "format": "tokenwise_pca_v1",
        "input_dim": str(mean.numel()),
        "output_dim": str(components.shape[0]),
        "projection_sha256": digest,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"mean": mean, "components": components}, str(path), metadata=payload)
    return digest


def load_projection(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as reader:
        mean = reader.get_tensor("mean").float().contiguous()
        components = reader.get_tensor("components").float().contiguous()
        metadata = {str(k): str(v) for k, v in (reader.metadata() or {}).items()}
    if mean.ndim != 1 or components.ndim != 2 or components.shape[1] != mean.numel():
        raise ValueError(f"Invalid projection shapes: mean={tuple(mean.shape)}, components={tuple(components.shape)}")
    actual = projection_sha256(mean, components)
    expected = metadata.get("projection_sha256")
    if expected and expected != actual:
        raise ValueError(f"Projection hash mismatch in {path}: {expected} != {actual}")
    return mean, components, metadata


def _transform_tensor(features: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3 or features.shape[-1] != mean.numel():
        raise ValueError(f"Feature shape {tuple(features.shape)} does not match projection input {mean.numel()}")
    flat = features.float().reshape(-1, features.shape[-1])
    projected = (flat - mean) @ components.T
    return projected.reshape(features.shape[0], features.shape[1], components.shape[0]).to(torch.bfloat16).contiguous()


def transform_tree(
    input_root: Path,
    output_root: Path,
    projection_path: Path,
    feature_key: str = "features",
    overwrite: bool = False,
    directory_name: str | None = None,
) -> dict[str, int | str]:
    mean, components, projection_meta = load_projection(projection_path)
    paths = episode_paths(input_root, directory_name)
    transformed = 0
    total_input_bytes = 0
    total_output_bytes = 0
    for path in paths:
        relative = path.relative_to(input_root)
        output_path = output_root / relative
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        with safe_open(path, framework="pt", device="cpu") as reader:
            tensors = {}
            for key in reader.keys():
                if key != feature_key:
                    tensors[key] = reader.get_tensor(key).contiguous()
                    continue
                tensors[key] = _transform_tensor(reader.get_tensor(feature_key), mean, components)
            metadata = {str(k): str(v) for k, v in (reader.metadata() or {}).items()}
        metadata.update({
            "feature_transform": "tokenwise_pca_v1",
            "projection_sha256": projection_meta.get("projection_sha256", projection_sha256(mean, components)),
            "projection_input_dim": str(mean.numel()),
            "projection_output_dim": str(components.shape[0]),
            "source_feature_dim": str(mean.numel()),
            "feature_dim": str(components.shape[0]),
            "source_feature_shape": json.dumps(list(tensors[feature_key].shape[:2]) + [mean.numel()]),
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        save_file(tensors, str(temporary), metadata=metadata)
        temporary.replace(output_path)
        transformed += 1
        total_input_bytes += path.stat().st_size
        total_output_bytes += output_path.stat().st_size
    return {
        "episodes": transformed,
        "input_bytes": total_input_bytes,
        "output_bytes": total_output_bytes,
        "projection_sha256": projection_meta.get("projection_sha256", projection_sha256(mean, components)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--input-root", type=Path, required=True)
    fit.add_argument("--projection", type=Path, required=True)
    fit.add_argument("--feature-key", default="features")
    fit.add_argument("--output-dim", type=int, required=True)
    fit.add_argument("--sample-rows", type=int, default=100_000)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--metadata", type=Path, default=None)
    fit.add_argument("--directory-name", default=None)
    fit.add_argument("--sampling-mode", choices=("global", "episode-balanced"), default="global")
    fit.add_argument("--anchors-per-episode", type=int, default=2)

    transform = subparsers.add_parser("transform")
    transform.add_argument("--input-root", type=Path, required=True)
    transform.add_argument("--output-root", type=Path, required=True)
    transform.add_argument("--projection", type=Path, required=True)
    transform.add_argument("--feature-key", default="features")
    transform.add_argument("--overwrite", action="store_true")
    transform.add_argument("--directory-name", default=None)

    fit_transform = subparsers.add_parser("fit-transform")
    for action in ("input_root", "projection"):
        fit_transform.add_argument(f"--{action.replace('_', '-')}", type=Path, required=True)
    fit_transform.add_argument("--output-root", type=Path, required=True)
    fit_transform.add_argument("--feature-key", default="features")
    fit_transform.add_argument("--output-dim", type=int, required=True)
    fit_transform.add_argument("--sample-rows", type=int, default=100_000)
    fit_transform.add_argument("--seed", type=int, default=42)
    fit_transform.add_argument("--metadata", type=Path, default=None)
    fit_transform.add_argument("--overwrite", action="store_true")
    fit_transform.add_argument("--directory-name", default=None)
    fit_transform.add_argument("--sampling-mode", choices=("global", "episode-balanced"), default="global")
    fit_transform.add_argument("--anchors-per-episode", type=int, default=2)
    args = parser.parse_args()

    if args.command in {"fit", "fit-transform"}:
        metadata = {}
        if args.metadata is not None:
            metadata = {str(k): str(v) for k, v in json.loads(args.metadata.read_text()).items()}
        mean, components = fit_projection(
            args.input_root, args.feature_key, args.output_dim, args.sample_rows, args.seed,
            args.directory_name, args.sampling_mode, args.anchors_per_episode,
        )
        metadata.update({
            "sampling_mode": args.sampling_mode,
            "sample_rows": str(args.sample_rows),
            "sampling_seed": str(args.seed),
        })
        if args.sampling_mode == "episode-balanced":
            metadata["anchors_per_episode"] = str(args.anchors_per_episode)
        digest = save_projection(args.projection, mean, components, metadata)
        print(json.dumps({"projection": str(args.projection), "projection_sha256": digest}, sort_keys=True))
        if args.command == "fit":
            return
    result = transform_tree(
        args.input_root, args.output_root, args.projection, args.feature_key,
        args.overwrite, args.directory_name,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
