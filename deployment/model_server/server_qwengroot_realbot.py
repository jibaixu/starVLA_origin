#!/usr/bin/env python3
"""Serve a joint-absolute RealBot QwenGR00T checkpoint over Torch/ZMQ.

The robot client sends raw RGB observations and raw joint positions.  This
server owns every model-specific operation: training-time image preprocessing,
state normalization, model inference, and action unnormalization.  Returned
``actions`` are physical absolute joint targets and are safe to interpret only
after the client validates the ``ping`` contract.

python deployment/model_server/server_qwengroot_realbot.py \
    --checkpoint playground/Checkpoints/starVLA/qwen3vl4b_gr00t_realbot_joint_abs/starvla_qwengroot_realbot_joint_abs/checkpoints/steps_50000_pytorch_model.pt \
    --host 127.0.0.1 \
    --port 35454 \
    --device cuda
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
import traceback
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import zmq

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import read_mode_config


LOGGER = logging.getLogger(__name__)
PROTOCOL = "starvla.realbot.torch_zmq.v2"
ACTION_SPACE = "joint_abs"
ACTION_VALUE_SPACE = "physical"
JOINT_ORDER = [f"joint_{index}" for index in range(1, 7)] + ["gripper"]
CAMERA_ORDER = ["image", "wrist_image"]
RAW_IMAGE_SHAPE = (480, 640, 3)
DEFAULT_TASKS = [
    "Flod_the_pink_towel",
    "erase_whiteboard",
    "pull_the_first_drawer",
    "stack_the_blue_cup_on_the_green_cup",
    "yellow_ball_blue_box",
]
# Stable dataset keys sent by clients -> exact language strings used in training.
TASK_LANGUAGE = {
    "Flod_the_pink_towel": "Flod the pink towel",
    "erase_whiteboard": "erase the whiteboard",
    # Preserve the source metadata typo for checkpoint reproducibility.
    "pull_the_first_drawer": "Flod the pink towel",
    "stack_the_blue_cup_on_the_green_cup": "stack the blue cup on the green cup",
    "yellow_ball_blue_box": "push the yellow ball into the blue box",
}


def letterbox_rgb(image: np.ndarray, target_size: tuple[int, int]) -> Image.Image:
    """Match RealBot training's neutral-gray letterbox preprocessing."""
    array = np.asarray(image)
    if array.shape != RAW_IMAGE_SHAPE or array.dtype != np.uint8:
        raise ValueError(
            f"Expected RGB uint8 image {RAW_IMAGE_SHAPE}, got {array.shape} {array.dtype}"
        )
    target_height, target_width = target_size
    source_height, source_width = array.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = Image.fromarray(array, mode="RGB").resize(
        (resized_width, resized_height), Image.Resampling.BILINEAR
    )
    canvas = Image.new("RGB", (target_width, target_height), color=(128, 128, 128))
    canvas.paste(
        resized,
        ((target_width - resized_width) // 2, (target_height - resized_height) // 2),
    )
    return canvas


def _serialize(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _deserialize(payload: bytes) -> Any:
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


class RealBotQwenGR00TPolicy:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: str | None = None,
        tasks: list[str] | None = None,
    ) -> None:
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        model_cfg, _ = read_mode_config(self.checkpoint)
        action_cfg = model_cfg["framework"]["action_model"]
        data_cfg = model_cfg["datasets"]["vla_data"]
        self.action_shape = (int(action_cfg["action_horizon"]), int(action_cfg["action_dim"]))
        self.state_dim = int(action_cfg["state_dim"])
        if self.action_shape != (16, 7) or self.state_dim != 7:
            raise ValueError(
                f"RealBot joint-abs server requires action_shape=(16,7), state_dim=7; "
                f"got action_shape={self.action_shape}, state_dim={self.state_dim}"
            )
        if str(data_cfg.get("action_mode")) != "abs":
            raise ValueError(f"Checkpoint action_mode must be 'abs', got {data_cfg.get('action_mode')!r}")

        configured_size = data_cfg.get("sample_image_size") or data_cfg.get("obs_image_size")
        if not isinstance(configured_size, (list, tuple)) or len(configured_size) != 2:
            raise ValueError(f"Checkpoint has invalid image size: {configured_size!r}")
        self.model_image_size = (int(configured_size[0]), int(configured_size[1]))
        self.control_fps = 15
        self.tasks = list(tasks or DEFAULT_TASKS)
        unknown_tasks = [task for task in self.tasks if task not in TASK_LANGUAGE]
        if unknown_tasks:
            raise ValueError(
                f"No training language mapping for task(s) {unknown_tasks}; "
                f"available={sorted(TASK_LANGUAGE)}"
            )

        self.processor = PolicyNormProcessor(self.checkpoint, unnorm_key=unnorm_key)
        expected_state_keys = [f"state.{name}" for name in JOINT_ORDER]
        expected_action_keys = [f"action.{name}" for name in JOINT_ORDER]
        if self.processor.state_keys != expected_state_keys:
            raise ValueError(
                f"Checkpoint state order {self.processor.state_keys} != {expected_state_keys}"
            )
        if self.processor.action_keys != expected_action_keys:
            raise ValueError(
                f"Checkpoint action order {self.processor.action_keys} != {expected_action_keys}"
            )

        action_stats = self.processor.action_statistics
        self.action_min = np.asarray(action_stats["min"], dtype=np.float32)
        self.action_max = np.asarray(action_stats["max"], dtype=np.float32)
        if self.action_min.shape != (7,) or self.action_max.shape != (7,):
            raise ValueError("Checkpoint action min/max must have shape [7]")

        LOGGER.info("Loading QwenGR00T checkpoint %s", self.checkpoint)
        model = baseframework.from_pretrained(self.checkpoint)
        if use_bf16:
            model = model.to(torch.bfloat16)
        self.model = model.to(device).eval()

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "protocol": PROTOCOL,
            "checkpoint": self.checkpoint,
            "action_space": ACTION_SPACE,
            "action_value_space": ACTION_VALUE_SPACE,
            "action_shape": list(self.action_shape),
            "state_shape": [self.state_dim],
            "state_order": list(JOINT_ORDER),
            "action_order": list(JOINT_ORDER),
            "camera_order": list(CAMERA_ORDER),
            "raw_image_shape": list(RAW_IMAGE_SHAPE),
            "model_image_size": list(self.model_image_size),
            "image_resize_mode": "letterbox",
            "control_fps": self.control_fps,
            "unnorm_key": self.processor.unnorm_key,
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
            "tasks": list(self.tasks),
            "task_languages": {task: TASK_LANGUAGE[task] for task in self.tasks},
        }

    def ping(self) -> dict[str, Any]:
        return self.contract

    def reset(self) -> dict[str, bool]:
        return {"ok": True}

    def get_action(self, observation: Mapping[str, Any], request_id: str) -> dict[str, Any]:
        task = observation.get("task")
        if not isinstance(task, str) or task not in self.tasks:
            raise ValueError(f"Unsupported task {task!r}; available tasks: {self.tasks}")

        raw_state = np.asarray(observation.get("observation.state"), dtype=np.float32)
        if raw_state.shape != (7,) or not np.isfinite(raw_state).all():
            raise ValueError(f"observation.state must be finite shape [7], got {raw_state.shape}")
        normalized_state = self.processor.apply_state(raw_state)

        images = []
        for camera in CAMERA_ORDER:
            key = f"observation.images.{camera}"
            if key not in observation:
                raise KeyError(f"Missing camera input {key!r}")
            images.append(letterbox_rgb(observation[key], self.model_image_size))

        example = {"image": images, "lang": TASK_LANGUAGE[task], "state": normalized_state}
        with torch.inference_mode():
            output = self.model.predict_action([example])
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        if normalized.shape != (1, *self.action_shape):
            raise ValueError(
                f"Model returned normalized_actions {normalized.shape}, expected {(1, *self.action_shape)}"
            )
        normalized = normalized[0]
        if not np.isfinite(normalized).all():
            raise ValueError("Model returned NaN or Inf actions")

        actions = np.asarray(self.processor.unapply_actions(normalized), dtype=np.float32)
        if actions.shape != self.action_shape or not np.isfinite(actions).all():
            raise ValueError(f"Unnormalized actions are invalid: {actions.shape}")
        return {
            "request_id": request_id,
            "actions": torch.from_numpy(np.ascontiguousarray(actions)),
            "normalized_actions": torch.from_numpy(np.ascontiguousarray(normalized)),
        }


class RealBotTorchZmqServer:
    def __init__(self, policy: RealBotQwenGR00TPolicy, host: str, port: int) -> None:
        self.policy = policy
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://{host}:{port}")

    def run(self) -> None:
        LOGGER.info("RealBot QwenGR00T server listening on %s", self.socket.getsockopt_string(zmq.LAST_ENDPOINT))
        try:
            while True:
                request = None
                try:
                    request = _deserialize(self.socket.recv())
                    if not isinstance(request, dict):
                        raise TypeError(f"Request must be a dict, got {type(request).__name__}")
                    endpoint = request.get("endpoint", "get_action")
                    request_id = str(request.get("request_id", ""))
                    if endpoint == "ping":
                        response = self.policy.ping()
                    elif endpoint == "reset":
                        response = self.policy.reset()
                    elif endpoint == "get_action":
                        response = self.policy.get_action(request.get("data", {}), request_id)
                    else:
                        raise ValueError(f"Unknown endpoint {endpoint!r}")
                    self.socket.send(_serialize(response))
                except Exception as exc:  # REP sockets must always reply.
                    LOGGER.error("Request failed: %s\n%s", exc, traceback.format_exc())
                    error = {
                        "status": "error",
                        "request_id": request.get("request_id", "") if isinstance(request, dict) else "",
                        "error": str(exc),
                    }
                    self.socket.send(_serialize(error))
        finally:
            self.socket.close(linger=0)
            self.context.term()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=45454)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--unnorm-key")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    policy = RealBotQwenGR00TPolicy(
        args.checkpoint,
        device=args.device,
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
        tasks=args.tasks,
    )
    RealBotTorchZmqServer(policy, args.host, args.port).run()


if __name__ == "__main__":
    main()
