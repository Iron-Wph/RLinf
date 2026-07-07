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

import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image

mp.set_start_method("spawn", force=True)


def _select_env_cfg(cfg):
    if "env" in cfg and "eval" in cfg.env:
        return cfg.env.eval
    if "env" in cfg and "train" in cfg.env:
        return cfg.env.train
    return cfg.env


def _load_env_seeds(env_cfg, num_envs: int) -> list[int]:
    seeds_path = env_cfg.get("seeds_path", None)
    task_name = env_cfg.task_config.task_name
    if seeds_path is not None and os.path.exists(seeds_path):
        with open(seeds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        task_seed_data = data.get(task_name, {})
        seeds = task_seed_data.get("success_seeds", None)
        if seeds:
            return [int(seed) for seed in seeds[:num_envs]]

    seed = int(env_cfg.get("seed", 0))
    return [seed + env_id for env_id in range(num_envs)]


def _save_image(path: Path, image) -> None:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _image_keys(obs: dict) -> list[str]:
    keys = []
    for key, value in obs.items():
        if key == "instruction":
            continue
        if key.endswith("_image") or key in (
            "full_image",
            "left_wrist_image",
            "right_wrist_image",
        ):
            if value is not None:
                keys.append(key)
    return keys


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="robotwin_camera_debug_franka_single_adjusted",
)
def main(cfg) -> None:
    env_cfg = _select_env_cfg(cfg)
    output_dir = Path(cfg.debug.output_dir)
    num_envs = int(cfg.debug.get("num_envs", env_cfg.get("total_num_envs", 1) or 1))
    camera_names = list(cfg.debug.get("camera_names", []))

    os.environ["ASSETS_PATH"] = env_cfg.assets_path

    task_config = OmegaConf.to_container(env_cfg.task_config, resolve=True)
    task_config["debug_camera_names"] = camera_names

    from robotwin.envs.vector_env import VectorEnv

    env_seeds = _load_env_seeds(env_cfg, num_envs)
    env = VectorEnv(
        task_config=task_config,
        n_envs=num_envs,
        env_seeds=env_seeds,
    )
    try:
        env.reset(env_seeds=env_seeds)
        obs_list = env.get_obs()

        metadata = {
            "num_envs": num_envs,
            "env_seeds": env_seeds,
            "assets_path": env_cfg.assets_path,
            "task_config": task_config,
            "saved_images": [],
        }

        for env_id, obs in enumerate(obs_list):
            for key in _image_keys(obs):
                image_path = output_dir / f"env_{env_id:02d}" / f"{key}.png"
                _save_image(image_path, obs[key])
                metadata["saved_images"].append(str(image_path))
            metadata.setdefault("instructions", {})[str(env_id)] = obs.get(
                "instruction", None
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved RoboTwin camera debug images to: {output_dir}")
        print(f"Metadata: {metadata_path}")
    finally:
        env.close(clear_cache=True)


if __name__ == "__main__":
    main()
