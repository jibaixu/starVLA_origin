from examples.simBenchmarks.RoboDojo.train_files.data_registry.data_config import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import (
    EMBODIMENT_TAG_MAPPING,
    EmbodimentTag,
)


def test_robodojo_registry_contract():
    robot_type = "robodojo_arx_x5_h50_q99"
    config = ROBOT_TYPE_CONFIG_MAP[robot_type]

    assert EmbodimentTag.ARX_X5.value == "arx_x5"
    assert EMBODIMENT_TAG_MAPPING["arx_x5"] == 9
    assert DATASET_NAMED_MIXTURES["robodojo_v21_all_h50_q99"] == [
        ("RoboDojo_lerobot_v21_video", 1.0, robot_type)
    ]
    assert config.action_indices == list(range(50))
    assert config.state_key_dims["state.left_joints"] == 6
    assert config.action_key_dims["action.right_gripper"] == 1
