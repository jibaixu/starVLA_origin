import importlib.util
import unittest

import numpy as np

from deployment.model_server.server_qwengroot_realbot import (
    DEFAULT_TASKS,
    TASK_LANGUAGE,
    letterbox_rgb,
)


def _load_client_module():
    path = "playground/realbot-master/deploy/deploy_for_arx5_qwengroot_joint_abs.py"
    spec = importlib.util.spec_from_file_location("realbot_joint_abs_client", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RealBotJointAbsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = _load_client_module()

    def test_letterbox_preserves_camera_geometry(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        output = np.asarray(letterbox_rgb(image, (224, 224)))
        self.assertEqual(output.shape, (224, 224, 3))
        # 4:3 input in a square canvas leaves top/bottom neutral-gray bars.
        np.testing.assert_array_equal(output[0, 0], [128, 128, 128])
        self.assertEqual(output[28, 0, 0], 255)

    def test_build_payload_selects_only_training_seven_dim_state(self):
        observation = {
            **{f"joint_{i}.pos": float(i) for i in range(1, 7)},
            "gripper.pos": 0.25,
            **{key: 0.0 for key in self.client.EEF_STATE_KEYS[:-1]},
            "image": np.zeros(self.client.IMAGE_SHAPE, dtype=np.uint8),
            "wrist_image": np.zeros(self.client.IMAGE_SHAPE, dtype=np.uint8),
        }
        payload = self.client.build_observation_payload(observation, "erase_whiteboard")
        np.testing.assert_array_equal(payload["observation.state"], [1, 2, 3, 4, 5, 6, 0.25])
        self.assertEqual(payload["observation.images.image"].shape, self.client.IMAGE_SHAPE)

    def test_action_validator_rejects_large_jump(self):
        policy = type(
            "Policy",
            (),
            {
                "action_min": np.full(7, -2.0, dtype=np.float32),
                "action_max": np.full(7, 2.0, dtype=np.float32),
            },
        )()
        actions = np.zeros((16, 7), dtype=np.float32)
        actions[0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "jump too large"):
            self.client.validate_action_chunk(
                actions,
                np.zeros(7, dtype=np.float32),
                policy,
                range_margin=0.05,
                max_joint_step=0.35,
                max_gripper_step=0.04,
            )

    def test_task_keys_match_training_language(self):
        expected = {
            "Flod_the_pink_towel": "Flod the pink towel",
            "erase_whiteboard": "erase the whiteboard",
            "pull_the_first_drawer": "Flod the pink towel",
            "stack_the_blue_cup_on_the_green_cup": "stack the blue cup on the green cup",
            "yellow_ball_blue_box": "push the yellow ball into the blue box",
        }
        self.assertEqual(TASK_LANGUAGE, expected)
        self.assertEqual(set(DEFAULT_TASKS), set(TASK_LANGUAGE))


if __name__ == "__main__":
    unittest.main()
