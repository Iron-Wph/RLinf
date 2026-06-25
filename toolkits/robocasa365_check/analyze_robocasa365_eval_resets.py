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

"""Analyze reset records emitted by a real RoboCasa365 evaluation run."""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any

EnvKey = tuple[Any, Any, int]


def _env_key(record: dict[str, Any], reset: dict[str, Any]) -> EnvKey:
    return (
        record.get("worker_rank"),
        record.get("seed_offset"),
        int(reset["env_id"]),
    )


def _load_resets(log_dir: pathlib.Path) -> dict[EnvKey, list[dict[str, Any]]]:
    resets_by_env: dict[EnvKey, list[dict[str, Any]]] = collections.defaultdict(list)
    log_paths = sorted(log_dir.glob("*.jsonl"))
    if not log_paths:
        raise FileNotFoundError(f"目录中没有 JSONL 日志: {log_dir}")

    for log_path in log_paths:
        with log_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{log_path}:{line_number} 不是有效 JSON: {exc}"
                    ) from exc
                if record.get("event") != "reset":
                    continue
                for reset in record.get("resets", []):
                    item = dict(reset)
                    item["_log_path"] = str(log_path)
                    item["_time"] = record.get("time")
                    resets_by_env[_env_key(record, reset)].append(item)

    if not resets_by_env:
        raise ValueError(
            f"{log_dir} 中没有 reset 事件。请确认 env.eval.debug_env_init.enabled=true。"
        )
    for resets in resets_by_env.values():
        resets.sort(key=lambda item: (int(item["reset_count"]), item["_time"] or 0))
    return dict(resets_by_env)


def _format_env_key(key: EnvKey) -> str:
    worker_rank, seed_offset, env_id = key
    return f"worker={worker_rank}, seed_offset={seed_offset}, env_id={env_id}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查真实 RoboCasa365 eval 的任务固定性和 reset 场景多样性。"
    )
    parser.add_argument("log_dir", type=pathlib.Path)
    parser.add_argument(
        "--expected-resets-per-env",
        type=int,
        default=None,
        help="通常等于 algorithm.eval_rollout_epoch。",
    )
    parser.add_argument(
        "--expected-num-tasks",
        type=int,
        default=None,
        help="期望实际分配到 env 的任务数量。",
    )
    parser.add_argument(
        "--expected-envs-per-task",
        type=int,
        default=None,
        help="并行评估时通常等于 total_num_envs / num_tasks。",
    )
    args = parser.parse_args()

    resets_by_env = _load_resets(args.log_dir)
    failures: list[str] = []
    task_to_envs: dict[tuple[int, str], set[EnvKey]] = collections.defaultdict(set)
    total_resets = 0
    total_adjacent_scene_pairs = 0
    changed_adjacent_scene_pairs = 0

    for key, resets in sorted(resets_by_env.items(), key=lambda item: str(item[0])):
        total_resets += len(resets)
        tasks = {
            (int(reset["task_id"]), str(reset["task_name"])) for reset in resets
        }
        if len(tasks) != 1:
            failures.append(
                f"{_format_env_key(key)} 在不同 episode 间切换了任务: {sorted(tasks)}"
            )
            continue

        task = next(iter(tasks))
        task_to_envs[task].add(key)
        reset_counts = [int(reset["reset_count"]) for reset in resets]
        expected_counts = list(range(reset_counts[0], reset_counts[0] + len(resets)))
        if reset_counts != expected_counts:
            failures.append(
                f"{_format_env_key(key)} reset_count 不连续: {reset_counts}"
            )

        if (
            args.expected_resets_per_env is not None
            and len(resets) != args.expected_resets_per_env
        ):
            failures.append(
                f"{_format_env_key(key)} reset 次数={len(resets)}，"
                f"期望={args.expected_resets_per_env}"
            )

        for previous, current in zip(resets, resets[1:]):
            total_adjacent_scene_pairs += 1
            previous_scene = previous.get("scene_sha256")
            current_scene = current.get("scene_sha256")
            if not previous_scene or not current_scene:
                failures.append(
                    f"{_format_env_key(key)} 缺少 scene_sha256，请使用包含场景指纹的代码重新评估。"
                )
                continue
            if previous_scene == current_scene:
                failures.append(
                    f"{_format_env_key(key)} 的相邻 episode 场景相同: "
                    f"reset_count={previous['reset_count']} -> {current['reset_count']}, "
                    f"scene_sha256={current_scene}"
                )
            else:
                changed_adjacent_scene_pairs += 1

    if (
        args.expected_num_tasks is not None
        and len(task_to_envs) != args.expected_num_tasks
    ):
        failures.append(
            f"实际任务数={len(task_to_envs)}，期望任务数={args.expected_num_tasks}"
        )

    if args.expected_envs_per_task is not None:
        for task, env_keys in sorted(task_to_envs.items()):
            if len(env_keys) != args.expected_envs_per_task:
                failures.append(
                    f"任务 {task[1]}(id={task[0]}) 分配 env 数={len(env_keys)}，"
                    f"期望={args.expected_envs_per_task}"
                )

    print("RoboCasa365 真实 eval reset 分析")
    print(f"  日志目录: {args.log_dir}")
    print(f"  env 数量: {len(resets_by_env)}")
    print(f"  reset 总数: {total_resets}")
    print(f"  实际任务数: {len(task_to_envs)}")
    print(
        "  相邻场景变化: "
        f"{changed_adjacent_scene_pairs}/{total_adjacent_scene_pairs}"
    )
    print("  每任务 env 数:")
    for (task_id, task_name), env_keys in sorted(task_to_envs.items()):
        print(f"    {task_name}(id={task_id}): {len(env_keys)}")

    if failures:
        print("\n检查失败:")
        for failure in failures:
            print(f"  FAIL  {failure}")
        raise SystemExit(1)

    print("\n检查通过:")
    print("  PASS  每个 env 在多次 reset 后任务保持不变")
    print("  PASS  每个 env 的相邻 episode 场景均发生变化")
    if args.expected_resets_per_env is not None:
        print("  PASS  每个 env 的 reset 次数符合预期")
    if args.expected_num_tasks is not None:
        print("  PASS  实际任务数量符合预期")
    if args.expected_envs_per_task is not None:
        print("  PASS  每个任务分配的 env 数量符合预期")


if __name__ == "__main__":
    main()
