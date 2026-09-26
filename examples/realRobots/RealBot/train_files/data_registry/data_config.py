"""RealBot ARX5 single-arm absolute-joint data configuration."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RealBotARX5AbsoluteJointsDataConfig:
    """Six arm joints plus one absolute gripper value from two cameras."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.image", "video.wrist_image"]
    state_keys = [
        "state.joint_1",
        "state.joint_2",
        "state.joint_3",
        "state.joint_4",
        "state.joint_5",
        "state.joint_6",
        "state.gripper",
    ]
    action_keys = [
        "action.joint_1",
        "action.joint_2",
        "action.joint_3",
        "action.joint_4",
        "action.joint_5",
        "action.joint_6",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "min_max" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "min_max" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "realbot_arx5_abs_joints": RealBotARX5AbsoluteJointsDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "realbot_joint_abs": [
        ("Flod_the_pink_towel", 1.0, "realbot_arx5_abs_joints"),
        ("erase_whiteboard", 1.0, "realbot_arx5_abs_joints"),
        ("pull_the_first_drawer", 1.0, "realbot_arx5_abs_joints"),
        ("stack_the_blue_cup_on_the_green_cup", 1.0, "realbot_arx5_abs_joints"),
        ("yellow_ball_blue_box", 1.0, "realbot_arx5_abs_joints"),
    ],
    "realbot_fit_flod_pink_towel": [("Flod_the_pink_towel", 1.0, "realbot_arx5_abs_joints")],
    "realbot_fit_erase_whiteboard": [("erase_whiteboard", 1.0, "realbot_arx5_abs_joints")],
    "realbot_fit_pull_first_drawer": [("pull_the_first_drawer", 1.0, "realbot_arx5_abs_joints")],
    "realbot_fit_stack_blue_cup": [("stack_the_blue_cup_on_the_green_cup", 1.0, "realbot_arx5_abs_joints")],
    "realbot_fit_yellow_ball": [("yellow_ball_blue_box", 1.0, "realbot_arx5_abs_joints")],
}
