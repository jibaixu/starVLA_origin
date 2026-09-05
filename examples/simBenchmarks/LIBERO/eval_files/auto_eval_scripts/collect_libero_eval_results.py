#!/usr/bin/env python3
"""Collect per-suite LIBERO result sidecars into one checkpoint summary."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def checkpoint_id(checkpoint: str | Path) -> str:
    """Return the stable result-file identifier for a model checkpoint."""
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.parent.name == "final_model":
        return "final_model"

    filename = checkpoint_path.name
    for suffix in ("_pytorch_model.pt", "_model.safetensors", ".pt", ".safetensors"):
        if filename.endswith(suffix):
            filename = filename[: -len(suffix)]
            break
    if filename.startswith("steps_") and filename[len("steps_") :].isdigit():
        return filename[len("steps_") :]
    return filename


def model_root(checkpoint: str | Path) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    if checkpoint_path.parent.name == "final_model":
        return checkpoint_path.parent.parent
    raise ValueError(
        "Checkpoint must be in '<run>/checkpoints/' or '<run>/final_model/': "
        f"{checkpoint_path}"
    )


def suite_result_path(checkpoint: str | Path, task_suite: str) -> Path:
    root = model_root(checkpoint)
    return root / "logs" / ".suite_results" / f"{checkpoint_id(checkpoint)}_{task_suite}.json"


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, target)


def collect_results(
    checkpoint: str | Path,
    task_suites: Iterable[str],
    result_timestamp: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write and return the aggregate result for one checkpoint."""
    checkpoint_path = Path(checkpoint)
    requested_suites = list(task_suites)
    suites: dict[str, Any] = {}
    missing_suites: list[str] = []
    invalid_suites: dict[str, str] = {}

    for task_suite in requested_suites:
        path = suite_result_path(checkpoint_path, task_suite)
        if not path.is_file():
            missing_suites.append(task_suite)
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                suite_result = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            invalid_suites[task_suite] = str(exc)
            continue

        if suite_result.get("checkpoint") != str(checkpoint_path):
            invalid_suites[task_suite] = "checkpoint does not match requested checkpoint"
            continue
        if suite_result.get("task_suite") != task_suite:
            invalid_suites[task_suite] = "task_suite does not match sidecar filename"
            continue
        successes = int(suite_result.get("total_successes", 0))
        episodes = int(suite_result.get("total_episodes", 0))
        suites[task_suite] = {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes if episodes else None,
        }

    total_successes = sum(result["successes"] for result in suites.values())
    total_episodes = sum(result["episodes"] for result in suites.values())
    status = "completed" if not missing_suites and not invalid_suites else "partial"
    evaluation = next((result.get("evaluation", {}) for result in suites.values()), {})
    result_timestamp = result_timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    summary: dict[str, Any] = {
        "status": status,
        "checkpoint": str(checkpoint_path),
        "checkpoint_id": checkpoint_id(checkpoint_path),
        "result_timestamp": result_timestamp,
        "evaluation": evaluation,
        "requested_task_suites": requested_suites,
        "suites": suites,
        "overall": {
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes if total_episodes else None,
        },
    }
    if missing_suites:
        summary["missing_suites"] = missing_suites
    if invalid_suites:
        summary["invalid_suites"] = invalid_suites

    output_path = (
        model_root(checkpoint_path)
        / "logs"
        / f"eval_result_{checkpoint_id(checkpoint_path)}_{result_timestamp}.json"
    )
    write_json_atomic(output_path, summary)
    return output_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-suite", action="append", dest="task_suites", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, summary = collect_results(args.checkpoint, args.task_suites)
    print(f"Wrote {summary['status']} LIBERO result summary: {output_path}")


if __name__ == "__main__":
    main()
