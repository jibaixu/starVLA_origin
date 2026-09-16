#!/usr/bin/env python3
"""Build a canonical episode manifest for RoboTwin teacher extraction.

The RoboTwin release stores each task as a separate LeRobot v2 dataset.  The
three teacher projects historically discover LIBERO suites only, so this small
adapter flattens all task roots into one deterministic JSONL manifest without
modifying the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _group_seed(seed: int, domain: str, task_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{domain}:{task_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _select_group_episodes(
    episodes: list[dict],
    *,
    domain: str,
    task_id: str,
    episodes_per_group: int | None,
    seed: int,
) -> list[dict]:
    if episodes_per_group is None:
        return episodes
    if episodes_per_group <= 0:
        raise ValueError("episodes_per_group must be positive")
    if len(episodes) < episodes_per_group:
        raise ValueError(
            f"Calibration group {domain}/{task_id} has {len(episodes)} episodes; "
            f"cannot select {episodes_per_group}"
        )
    ordered = sorted(episodes, key=lambda episode: int(episode["episode_index"]))
    rng = random.Random(_group_seed(seed, domain, task_id))
    selected_indices = set(rng.sample(range(len(ordered)), episodes_per_group))
    return [episode for index, episode in enumerate(ordered) if index in selected_indices]


def build(
    data_root: Path,
    split: str = "all",
    episodes_per_group: int | None = None,
    seed: int = 42,
) -> list[dict]:
    rows: list[dict] = []
    global_id = 0
    for domain in ("Clean", "Randomized"):
        if split not in ("all", domain.lower()):
            continue
        domain_root = data_root / domain
        for task_root in sorted(
            p for p in domain_root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / "meta" / "info.json").is_file()
        ):
            meta = task_root / "meta"
            info = _read_json(meta / "info.json")
            episodes = [
                json.loads(line)
                for line in (meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            chunks_size = int(info.get("chunks_size", 1000))
            video_pattern = str(info["video_path"])
            video_keys = list(info.get("features", {}))
            video_keys = [k for k in video_keys if str(info["features"][k].get("dtype")) == "video"]
            selected_episodes = _select_group_episodes(
                episodes,
                domain=domain.lower(),
                task_id=task_root.name,
                episodes_per_group=episodes_per_group,
                seed=seed,
            )
            for episode in selected_episodes:
                episode_index = int(episode["episode_index"])
                task = str(episode.get("tasks", [task_root.name])[0])
                videos = {}
                for key in video_keys:
                    rel = video_pattern.format(
                        episode_chunk=episode_index // chunks_size,
                        episode_index=episode_index,
                        video_key=key,
                    )
                    path = task_root / rel
                    if path.is_file():
                        videos[key] = str(path)
                if not videos:
                    raise FileNotFoundError(f"No videos found for {task_root} episode {episode_index}")
                preferred_video_key = (
                    "observation.images.cam_high"
                    if "observation.images.cam_high" in videos
                    else (
                        "observation.images.image"
                        if "observation.images.image" in videos
                        else next(iter(videos))
                    )
                )
                rows.append({
                    "manifest_id": global_id,
                    "episode_index": episode_index,
                    "task_id": task_root.name,
                    "suite": f"robotwin_{domain.lower()}_{task_root.name}",
                    "domain": domain.lower(),
                    "dataset_root": str(task_root.resolve()),
                    "episode_length": int(episode["length"]),
                    "length": int(episode["length"]),
                    "fps": int(info.get("fps", 15)),
                    "chunks_size": chunks_size,
                    "prompt": task,
                    "tasks": [task],
                    "videos": videos,
                    "video_path": videos[preferred_video_key],
                    "video_key": preferred_video_key,
                })
                global_id += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("all", "clean", "randomized"), default="all")
    parser.add_argument(
        "--episodes-per-group",
        type=int,
        default=None,
        help="Select this many episodes per domain/task group (for calibration).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = build(
        args.data_root.expanduser().resolve(),
        args.split,
        episodes_per_group=args.episodes_per_group,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} episodes to {args.output}")


if __name__ == "__main__":
    main()
