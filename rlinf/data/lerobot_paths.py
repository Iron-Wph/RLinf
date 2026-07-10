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

"""Helpers for passing LeRobot dataset paths to OpenPI."""

from typing import Any


def resolve_lerobot_repo_id(data_paths: Any) -> str | None:
    """Extract one local dataset path or LeRobot repo id from SFT config."""
    if data_paths is None:
        return None
    if isinstance(data_paths, str):
        return data_paths
    if isinstance(data_paths, dict):
        path = data_paths.get("dataset_path", data_paths.get("data_path"))
        return str(path) if path is not None else None
    if isinstance(data_paths, (list, tuple)):
        if not data_paths:
            return None
        first = data_paths[0]
        if isinstance(first, dict):
            path = first.get("dataset_path", first.get("data_path"))
            if path is None:
                raise ValueError(
                    "Each dataset entry must define 'dataset_path' or 'data_path'."
                )
            return str(path)
        return str(first)
    return str(data_paths)
