"""Plot a complete RealBot trajectory against non-overlapping model predictions.

Example:
  python examples/realRobots/RealBot/train_files/visualize_realbot_fit.py \
    --prediction-source server --task stack_the_blue_cup_on_the_green_cup --episode 17 \
    --output playground/visualizations/realbot_50k_server_stack_blue_cup_ep17.png

  python examples/realRobots/RealBot/train_files/visualize_realbot_fit.py \
    --checkpoint playground/Checkpoints/starVLA/qwen3vl4b_gr00t_realbot_joint_abs/starvla_qwengroot_realbot_joint_abs/checkpoints/steps_50000_pytorch_model.pt \
    --task Flod_the_pink_towel --episode 12 \
    --output playground/visualizations/realbot_flod_pink_towel_50k_ep12.png
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import random
from typing import Any
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zmq

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config


TASK_TO_MIX = {
    "Flod_the_pink_towel": "realbot_fit_flod_pink_towel",
    "erase_whiteboard": "realbot_fit_erase_whiteboard",
    "pull_the_first_drawer": "realbot_fit_pull_first_drawer",
    "stack_the_blue_cup_on_the_green_cup": "realbot_fit_stack_blue_cup",
    "yellow_ball_blue_box": "realbot_fit_yellow_ball",
}
ACTION_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
STATE_KEYS = [f"state.{name}" for name in ACTION_NAMES]
IMAGE_KEYS = ["video.image", "video.wrist_image"]
RAW_IMAGE_SHAPE = (480, 640, 3)
ACTION_SHAPE = (16, 7)
SERVER_PROTOCOL = "starvla.realbot.torch_zmq.v2"
ROBOT_TYPE = "realbot_arx5_abs_joints"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("playground/Checkpoints/starVLA/qwen3vl4b_gr00t_realbot_joint_abs/starvla_qwengroot_realbot_joint_abs/checkpoints/steps_20000_pytorch_model.pt"))
    parser.add_argument("--prediction-source", choices=("server", "local"), default="local")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=35454)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--task", default="auto", choices=["auto", *sorted(TASK_TO_MIX)])
    parser.add_argument("--episode", type=int, default=-1, help="Trajectory id; -1 selects one randomly")
    parser.add_argument("--data-root", type=Path, default=None, help="Override the dataset root from the checkpoint config")
    parser.add_argument("--output", required=True, help="PNG output path")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--stride", type=int, default=16, help="Prediction chunk stride")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _find_action_stats(norm_stats: dict) -> dict:
    """Find the flattened action statistics saved by LeRobotMixtureDataset."""
    if not isinstance(norm_stats, dict):
        raise ValueError("Checkpoint normalization statistics must be a dictionary")
    candidates = [norm_stats]
    candidates.extend(value for value in norm_stats.values() if isinstance(value, dict))
    for candidate in candidates:
        action = candidate.get("action")
        if isinstance(action, dict) and "min" in action and "max" in action:
            return action
        statistics = candidate.get("statistics")
        if isinstance(statistics, dict) and isinstance(statistics.get("action"), dict):
            action = statistics["action"]
            if "min" in action and "max" in action:
                return action
    raise KeyError("No flattened action min/max statistics found in checkpoint")


class TorchZmqPolicyClient:
    """Minimal client for server_qwengroot_realbot.py."""

    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{port}")
        self.session = uuid4().hex
        self.sequence = 0

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        self.sequence += 1
        request = {"endpoint": endpoint, "request_id": f"{self.session}-{self.sequence}"}
        if data is not None:
            request["data"] = data
        buffer = io.BytesIO()
        torch.save(request, buffer)
        self.socket.send(buffer.getvalue())
        response = torch.load(io.BytesIO(self.socket.recv()), map_location="cpu", weights_only=False)
        if isinstance(response, dict) and response.get("status") == "error":
            raise RuntimeError(f"Server {endpoint} failed: {response.get('error')}")
        return response

    def check(self, task: str) -> dict[str, Any]:
        contract = self.call("ping")
        checks = {
            "protocol": (contract.get("protocol"), SERVER_PROTOCOL),
            "action_space": (contract.get("action_space"), "joint_abs"),
            "action_value_space": (contract.get("action_value_space"), "physical"),
            "action_shape": (contract.get("action_shape"), list(ACTION_SHAPE)),
            "state_shape": (contract.get("state_shape"), [7]),
            "state_order": (contract.get("state_order"), ACTION_NAMES),
            "action_order": (contract.get("action_order"), ACTION_NAMES),
            "camera_order": (contract.get("camera_order"), ["image", "wrist_image"]),
            "raw_image_shape": (contract.get("raw_image_shape"), list(RAW_IMAGE_SHAPE)),
            "model_image_size": (contract.get("model_image_size"), [224, 224]),
            "image_resize_mode": (contract.get("image_resize_mode"), "letterbox"),
        }
        for name, (actual, expected) in checks.items():
            if actual != expected:
                raise ValueError(f"Server contract {name} mismatch: expected {expected!r}, got {actual!r}")
        if task not in contract.get("tasks", []):
            raise ValueError(f"Task {task!r} is not served; available={contract.get('tasks')}")
        task_languages = contract.get("task_languages", {})
        if not isinstance(task_languages.get(task), str) or not task_languages[task].strip():
            raise ValueError(f"Server has no language mapping for task {task!r}")
        return contract

    def predict(self, observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        response = self.call("get_action", observation)
        actions = response.get("actions")
        normalized = response.get("normalized_actions")
        actions = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions)
        normalized = normalized.detach().cpu().numpy() if isinstance(normalized, torch.Tensor) else np.asarray(normalized)
        actions = np.asarray(actions, dtype=np.float32)
        normalized = np.asarray(normalized, dtype=np.float32)
        if actions.shape != ACTION_SHAPE or normalized.shape != ACTION_SHAPE:
            raise ValueError(f"Server returned actions={actions.shape}, normalized={normalized.shape}")
        if not np.isfinite(actions).all() or not np.isfinite(normalized).all():
            raise ValueError("Server returned NaN or Inf actions")
        return actions, normalized

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def _latest_image(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[-1]
    if array.shape != RAW_IMAGE_SHAPE or array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 {RAW_IMAGE_SHAPE}, got {array.shape} {array.dtype}")
    return np.ascontiguousarray(array)


def _raw_server_observation(dataset, task: str, episode: int, frame: int) -> dict[str, Any]:
    raw = dataset.get_step_data(episode, frame)
    state = np.asarray([np.asarray(raw[key], dtype=np.float32).reshape(-1)[0] for key in STATE_KEYS], dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError(f"Non-finite state at frame {frame}")
    return {
        "observation.state": state,
        "observation.images.image": _latest_image(raw[IMAGE_KEYS[0]], IMAGE_KEYS[0]),
        "observation.images.wrist_image": _latest_image(raw[IMAGE_KEYS[1]], IMAGE_KEYS[1]),
        "task": task,
    }


def _inverse_min_max(values: np.ndarray, action_stats: dict) -> np.ndarray:
    low = np.asarray(action_stats["min"], dtype=np.float32)
    high = np.asarray(action_stats["max"], dtype=np.float32)
    if values.shape[-1] != low.shape[0] or low.shape != high.shape:
        raise ValueError(
            f"Action statistics shape mismatch: values={values.shape}, min={low.shape}, max={high.shape}"
        )
    return (values + 1.0) * 0.5 * (high - low) + low


def _build_sample(dataset, episode: int, frame: int) -> dict:
    raw = dataset.get_step_data(episode, frame)
    transformed = dataset.transforms(raw)
    return dataset._pack_sample(transformed, trajectory_id=episode, base_index=frame)


def main() -> None:
    args = _parse_args()
    if args.stride <= 0 or args.timeout_ms <= 0:
        raise ValueError("--stride must be positive")
    rng = random.Random(args.seed)
    task = rng.choice(sorted(TASK_TO_MIX)) if args.task == "auto" else args.task
    checkpoint = args.checkpoint.expanduser().resolve()

    if args.prediction_source == "local":
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Local QwenGR00T visualization requires CUDA")
        model = baseframework.from_pretrained(str(checkpoint)).to(args.device).eval()
        data_cfg = model.config.datasets.vla_data
    else:
        model = None
        config_dict, _ = read_mode_config(str(checkpoint))
        data_cfg = config_dict.get("datasets", {}).get("vla_data", {})

    if args.data_root is not None:
        data_root = args.data_root.expanduser().resolve()
    elif data_cfg is not None:
        configured_root = (
            data_cfg.get("data_root_dir")
            if isinstance(data_cfg, dict)
            else data_cfg.data_root_dir
        )
        data_root = Path(str(configured_root)).expanduser()
        if not data_root.is_absolute():
            data_root = (Path.cwd() / data_root).resolve()
    else:
        data_root = (Path.cwd() / "playground/Datasets/Realbot/joint_abs").resolve()

    dataset = make_LeRobotSingleDataset(
        data_root,
        task,
        ROBOT_TYPE,
        data_cfg=data_cfg,
    )
    available_episodes = [int(value) for value in dataset.trajectory_ids]
    episode = rng.choice(available_episodes) if args.episode < 0 else args.episode
    if episode not in set(available_episodes):
        raise ValueError(f"Episode {episode} is not present in {task}")

    trajectory = dataset.get_trajectory_data(episode)
    gt_actions = np.stack(trajectory["action"].to_numpy()).astype(np.float32)
    trajectory_length = gt_actions.shape[0]
    pred_actions = np.full_like(gt_actions, np.nan, dtype=np.float32)
    normalized_actions = np.full_like(gt_actions, np.nan, dtype=np.float32)
    server_contract = None
    client = None

    try:
        if args.prediction_source == "server":
            client = TorchZmqPolicyClient(args.server_host, args.server_port, args.timeout_ms)
            server_contract = client.check(task)
            client.call("reset")

        for start in range(0, trajectory_length, args.stride):
            if args.prediction_source == "server":
                physical_chunk, normalized_chunk = client.predict(
                    _raw_server_observation(dataset, task, episode, start)
                )
            else:
                sample = _build_sample(dataset, episode, start)
                with torch.inference_mode():
                    output = model.predict_action([sample])
                normalized_chunk = np.asarray(output["normalized_actions"])[0].astype(np.float32)
                physical_chunk = _inverse_min_max(normalized_chunk, _find_action_stats(model.norm_stats))
            end = min(start + args.stride, trajectory_length, start + physical_chunk.shape[0])
            pred_actions[start:end] = physical_chunk[: end - start]
            normalized_actions[start:end] = normalized_chunk[: end - start]
    finally:
        if client is not None:
            client.close()

    valid = np.isfinite(pred_actions).all(axis=1)
    if not valid.any():
        raise RuntimeError("No valid predictions were produced")
    errors = pred_actions[valid] - gt_actions[valid]
    mae = np.mean(np.abs(errors), axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))

    fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
    time_axis = np.arange(trajectory_length)
    for index, (axis, name) in enumerate(zip(axes, ACTION_NAMES)):
        axis.plot(time_axis, gt_actions[:, index], label="GT", color="#1f77b4", linewidth=1.5)
        axis.plot(time_axis, pred_actions[:, index], label="Pred", color="#d62728", linewidth=1.2)
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
        axis.set_title(f"{name} | MAE={mae[index]:.5f} RMSE={rmse[index]:.5f}", loc="left", fontsize=10)
    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"RealBot fit: {task} / episode {episode} | source={args.prediction_source} | stride={args.stride}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "task": task,
                "episode": episode,
                "checkpoint": str(checkpoint),
                "prediction_source": args.prediction_source,
                "server_contract": server_contract,
                "trajectory_length": trajectory_length,
                "stride": args.stride,
                "valid_prediction_frames": int(valid.sum()),
                "mae": dict(zip(ACTION_NAMES, mae.tolist())),
                "rmse": dict(zip(ACTION_NAMES, rmse.tolist())),
            },
            indent=2,
        )
    )
    npz_path = output_path.with_suffix(".npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        gt_actions=gt_actions,
        predicted_actions=pred_actions,
        normalized_actions=normalized_actions,
        task=np.asarray(task),
        episode=np.asarray(episode),
    )
    print(f"Saved plot to {output_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved arrays to {npz_path}")


if __name__ == "__main__":
    main()
