# Copyright 2025 The RLinf Authors.
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
# openpi model configs
import difflib

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)

from rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig import (
    LeRobotLiberoDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.metaworld_dataconfig import (
    LeRobotMetaworldDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.aloha_dataconfig import (
    LeRobotAlohaDataConfig
)

_CONFIGS = [
    TrainConfig(
        name="pi0_libero",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_metaworld",
        model=pi0_config.Pi0Config(action_horizon=5),
        data=LeRobotMetaworldDataConfig(
            repo_id="lerobot/metaworld_mt50",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_metaworld/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_metaworld",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=5, discrete_state_input=False
        ),
        data=LeRobotMetaworldDataConfig(
            repo_id="lerobot/metaworld_mt50",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_metaworld/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="place_empty_cup_random",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            # TODO: sft数据准备和sft之前都要修改这个
            repo_id="robotwin/place_empty_cup_random",  # your datasets repo_id
            # TODO: 为什么这个要设置为False？在openpi-main里默认是true？
            adapt_to_pi=True,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                # local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
            assets=AssetsConfig(
                assets_dir="/mnt/mnt/public/liuzhihao/RoboTwin-main/policy/pi0/assets/pi0_base_aloha_robotwin_full"
            ),
        ),
        freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        # run on J2
        pytorch_weight_path="/mnt/mnt/public/chenkang/openpi-main/checkpoints/torch/pi0_base",
        # debug on I
        # pytorch_weight_path="/mnt/mnt/public/liuzhihao/openpi-main/checkpoints/torch/pi0_base",
        # fsdp_devices=4,  # refer line 359
    ),
    TrainConfig(
        name="place_empty_cup_clean",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            # TODO: sft数据准备和sft之前都要修改这个
            repo_id="robotwin/place_empty_cup_clean",  # your datasets repo_id
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                # local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
            assets=AssetsConfig(
                assets_dir="/mnt/mnt/public/liuzhihao/RoboTwin-main/policy/pi0/assets/pi0_base_aloha_robotwin_full"
            ),
        ),
        freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        # run on J2
        pytorch_weight_path="/mnt/mnt/public/chenkang/openpi-main/checkpoints/torch/pi0_base",
        # debug on I
        # pytorch_weight_path="/mnt/mnt/public/liuzhihao/openpi-main/checkpoints/torch/pi0_base",
        # fsdp_devices=4,  # refer line 359
    ),
    TrainConfig(
        name="adjust_bottle_test",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            # TODO: sft数据准备和sft之前都要修改这个
            repo_id="test_repo_id",  # your datasets repo_id
            # TODO: 为什么这个要设置为False？在openpi-main里默认是true？
            adapt_to_pi=True,
            repack_transforms=_transforms.Group(inputs=[
                _transforms.RepackTransform({
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                })
            ]),
            base_config=DataConfig(
                # local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,  # Set to True for prompt by task_name
            ),
            assets=AssetsConfig(
                assets_dir="/mnt/mnt/public/liuzhihao/RoboTwin-main/policy/pi0/assets/pi0_base_aloha_robotwin_full"
            ),
        ),
        freeze_filter=pi0_config.Pi0Config().get_freeze_filter(),
        batch_size=32,  # the total batch_size not pre_gpu batch_size
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30000,
        # run on J2
        pytorch_weight_path="/mnt/mnt/public/chenkang/openpi-main/checkpoints/torch/pi0_base",
        # debug on I
        # pytorch_weight_path="/mnt/mnt/public/liuzhihao/openpi-main/checkpoints/torch/pi0_base",
        # fsdp_devices=4,  # refer line 359
    ),
]


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_openpi_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
