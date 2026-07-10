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
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[2] / "rlinf" / "data" / "lerobot_paths.py"
_SPEC = importlib.util.spec_from_file_location("lerobot_paths", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
resolve_lerobot_repo_id = _MODULE.resolve_lerobot_repo_id


@pytest.mark.parametrize(
    ("data_paths", "expected"),
    [
        ("/data/robotwin/franka", "/data/robotwin/franka"),
        ("robotwin/franka", "robotwin/franka"),
        (["/data/robotwin/franka"], "/data/robotwin/franka"),
        ({"dataset_path": "/data/robotwin/franka"}, "/data/robotwin/franka"),
        ([{"data_path": "/data/robotwin/franka"}], "/data/robotwin/franka"),
        (None, None),
    ],
)
def test_resolve_lerobot_repo_id(data_paths, expected):
    assert resolve_lerobot_repo_id(data_paths) == expected


def test_resolve_lerobot_repo_id_rejects_mapping_without_path():
    with pytest.raises(ValueError, match="dataset_path"):
        resolve_lerobot_repo_id([{"weight": 1.0}])
