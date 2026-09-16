#!/usr/bin/env python3
"""Split a RoboTwin JSONL manifest into deterministic balanced shards.

Episodes are assigned greedily by descending episode length so that each shard
has a similar estimated extraction cost. Every input row is preserved exactly
once in one output JSONL file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if "dataset_root" not in row or "episode_index" not in row or "length" not in row:
                raise ValueError(
                    f"Missing dataset_root/episode_index/length at {path}:{line_number}"
                )
            identity = (str(row["dataset_root"]), int(row["episode_index"]))
            if identity in seen:
                raise ValueError(f"Duplicate episode in manifest: {identity}")
            seen.add(identity)
            row["episode_index"] = int(row["episode_index"])
            row["length"] = int(row["length"])
            if row["length"] <= 0:
                raise ValueError(f"Episode length must be positive at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _tie_key(row: dict[str, Any], seed: int) -> str:
    identity = f"{seed}:{row['dataset_root']}:{int(row['episode_index'])}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def split_rows(
    rows: list[dict[str, Any]],
    num_shards: int,
    *,
    seed: int = 42,
) -> list[list[dict[str, Any]]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if len(rows) < num_shards:
        raise ValueError(f"Need at least {num_shards} episodes, got {len(rows)}")

    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    totals = [0] * num_shards
    ordered = sorted(
        rows,
        key=lambda row: (-int(row["length"]), _tie_key(row, seed)),
    )
    for row in ordered:
        shard_index = min(range(num_shards), key=lambda index: (totals[index], len(shards[index]), index))
        shards[shard_index].append(row)
        totals[shard_index] += int(row["length"])
    return shards


def write_shards(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    num_shards: int = 8,
    seed: int = 42,
    overwrite: bool = False,
    prefix: str = "robotwin_teacher_calibration_manifest_8pergroup",
) -> list[Path]:
    shards = split_rows(rows, num_shards, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, shard in enumerate(shards):
        path = output_dir / f"{prefix}_part{index:02d}.jsonl"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite")
        with path.open("w", encoding="utf-8") as handle:
            for row in sorted(shard, key=lambda item: (str(item["dataset_root"]), int(item["episode_index"]))):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="robotwin_teacher_calibration_manifest_8pergroup")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    paths = write_shards(
        rows,
        args.output_dir,
        num_shards=args.num_shards,
        seed=args.seed,
        overwrite=args.overwrite,
        prefix=args.prefix,
    )
    shards = split_rows(rows, args.num_shards, seed=args.seed)
    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "episodes": len(rows),
        "frames": sum(int(row["length"]) for row in rows),
        "shards": [
            {
                "path": str(path),
                "episodes": len(shard),
                "frames": sum(int(row["length"]) for row in shard),
            }
            for path, shard in zip(paths, shards)
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
