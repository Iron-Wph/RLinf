# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest

pytest.importorskip("openpi")

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config  # noqa: E402
from rlinf.models.embodiment.openpi.policies.robotwin_franka_policy import (  # noqa: E402
    RobotwinFrankaInputs,
    RobotwinFrankaOutputs,
)


def test_robotwin_franka_openpi_config_is_registered():
    config = get_openpi_config("pi0_base_franka_robotwin_full")

    assert config.data.repo_id == "robotwin/stack_blocks_two_franka_300"
    assert config.data.base_config.prompt_from_task is True
    assert config.model.action_horizon == 50

    data_config = config.data.create(config.assets_dirs, config.model)
    delta_transform = data_config.data_transforms.inputs[-1]
    assert tuple(delta_transform.mask) == (True,) * 7 + (False,)


def test_robotwin_franka_inputs_pad_qpos_and_mask_missing_right_wrist():
    transform = RobotwinFrankaInputs(action_dim=32)
    state = np.arange(8, dtype=np.float32)
    actions = np.arange(50 * 8, dtype=np.float32).reshape(50, 8)
    base_image = np.ones((3, 12, 16), dtype=np.uint8)
    wrist_image = np.full((3, 12, 16), 2, dtype=np.uint8)

    result = transform(
        {
            "images": {
                "cam_high": base_image,
                "cam_left_wrist": wrist_image,
            },
            "state": state,
            "actions": actions,
            "prompt": b"stack the blocks",
        }
    )

    assert result["state"].shape == (32,)
    np.testing.assert_array_equal(result["state"][:8], state)
    np.testing.assert_array_equal(result["state"][8:], 0)
    assert result["actions"].shape == (50, 32)
    np.testing.assert_array_equal(result["actions"][:, :8], actions)
    np.testing.assert_array_equal(result["actions"][:, 8:], 0)
    assert result["image"]["base_0_rgb"].shape == (12, 16, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (12, 16, 3)
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], 0)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert result["prompt"] == "stack the blocks"


def test_robotwin_franka_outputs_remove_openpi_padding():
    actions = np.arange(50 * 32, dtype=np.float32).reshape(50, 32)
    result = RobotwinFrankaOutputs()({"actions": actions})
    np.testing.assert_array_equal(result["actions"], actions[:, :8])


def test_robotwin_franka_inputs_accept_rlinf_environment_observation():
    transform = RobotwinFrankaInputs(action_dim=32)
    result = transform(
        {
            "observation/image": np.ones((12, 16, 3), dtype=np.uint8),
            "observation/wrist_image": np.ones((1, 12, 16, 3), dtype=np.uint8),
            "observation/state": np.arange(8, dtype=np.float32),
            "prompt": "stack the blocks",
        }
    )

    assert result["state"].shape == (32,)
    assert result["image"]["base_0_rgb"].shape == (12, 16, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (12, 16, 3)
    assert result["image_mask"]["right_wrist_0_rgb"] == np.False_


@pytest.mark.parametrize(
    ("state_shape", "action_shape"),
    [((7,), (50, 8)), ((8,), (50, 14))],
)
def test_robotwin_franka_inputs_reject_wrong_dimensions(state_shape, action_shape):
    transform = RobotwinFrankaInputs(action_dim=32)
    sample = {
        "images": {
            "cam_high": np.zeros((3, 12, 16), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 12, 16), dtype=np.uint8),
        },
        "state": np.zeros(state_shape, dtype=np.float32),
        "actions": np.zeros(action_shape, dtype=np.float32),
    }

    with pytest.raises(ValueError):
        transform(sample)
