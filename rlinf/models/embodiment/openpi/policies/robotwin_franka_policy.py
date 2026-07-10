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
import dataclasses
from typing import ClassVar

import einops
import numpy as np
from openpi import transforms

FRANKA_ACTION_DIM = 8


def _convert_image_to_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _decode_robotwin_franka(data: dict) -> dict:
    if "observation/state" not in data:
        return data

    if "observation/image" not in data:
        raise ValueError("Missing required RoboTwin Franka observation/image")
    if "observation/wrist_image" not in data:
        raise ValueError("Missing required RoboTwin Franka observation/wrist_image")

    wrist_images = np.asarray(data["observation/wrist_image"])
    if wrist_images.ndim == 4:
        if wrist_images.shape[0] != 1:
            raise ValueError(
                f"Expected one single-arm wrist image, got shape {wrist_images.shape}"
            )
        wrist_image = wrist_images[0]
    elif wrist_images.ndim == 3:
        wrist_image = wrist_images
    else:
        raise ValueError(
            "Expected wrist image shape (H, W, C), (C, H, W), or one leading "
            f"camera dimension, got {wrist_images.shape}"
        )

    decoded = dict(data)
    decoded["state"] = data["observation/state"]
    decoded["images"] = {
        "cam_high": data["observation/image"],
        "cam_left_wrist": wrist_image,
    }
    return decoded


@dataclasses.dataclass(frozen=True)
class RobotwinFrankaInputs(transforms.DataTransformFn):
    """Convert single-arm RoboTwin Franka qpos samples to OpenPI inputs."""

    action_dim: int

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        data = _decode_robotwin_franka(data)
        if self.action_dim < FRANKA_ACTION_DIM:
            raise ValueError(
                f"OpenPI action_dim must be at least {FRANKA_ACTION_DIM}, "
                f"got {self.action_dim}"
            )

        state = np.asarray(data["state"])
        if state.shape != (FRANKA_ACTION_DIM,):
            raise ValueError(
                "Expected RoboTwin Franka state shape "
                f"({FRANKA_ACTION_DIM},), got {state.shape}"
            )
        state = transforms.pad_to_dim(state, self.action_dim)

        in_images = data["images"]
        unexpected_cameras = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected_cameras:
            raise ValueError(
                f"Expected cameras from {self.EXPECTED_CAMERAS}, "
                f"got unexpected cameras {tuple(sorted(unexpected_cameras))}"
            )
        missing_cameras = {"cam_high", "cam_left_wrist"} - set(in_images)
        if missing_cameras:
            raise ValueError(
                f"Missing required RoboTwin Franka cameras: {tuple(sorted(missing_cameras))}"
            )

        base_image = _convert_image_to_hwc(in_images["cam_high"])
        wrist_image = _convert_image_to_hwc(in_images["cam_left_wrist"])
        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[-1] != FRANKA_ACTION_DIM:
                raise ValueError(
                    "Expected RoboTwin Franka actions shape "
                    f"(N, {FRANKA_ACTION_DIM}), got {actions.shape}"
                )
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class RobotwinFrankaOutputs(transforms.DataTransformFn):
    """Remove OpenPI action padding for the 8D RoboTwin Franka controller."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :FRANKA_ACTION_DIM])}
