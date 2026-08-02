#!/usr/bin/env python3
"""Validate the RoboTwin pi0 ALOHA LoRA recipe and its OpenPI LR schedule."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(
    "/mnt/public2/wph/codes/develop_async/RoboTwin_main_official/data/robotwin/adjust_bottle"
)
DEFAULT_ASSETS = Path(
    "/mnt/public2/wph/models/pi0_base/pi0_base/physical-intelligence/robotwin/adjust_bottle"
)


def _expect_contains(path: Path, expected: list[str], failures: list[str]) -> None:
    content = path.read_text(encoding="utf-8")
    for fragment in expected:
        if fragment not in content:
            failures.append(f"{path}: missing {fragment!r}")


def check_recipe(dataset: Path, assets: Path) -> list[str]:
    """Check the fixed pi0/ALOHA/LoRA and local-data integration contract."""
    failures: list[str] = []
    model_recipe = REPO_ROOT / "examples/sft/config/model/pi0_robotwin_lora.yaml"
    train_recipe = REPO_ROOT / "examples/sft/config/robotwin_pi0_aloha_lora.yaml"
    for path in (model_recipe, train_recipe):
        if not path.is_file():
            failures.append(f"missing recipe: {path}")
    if failures:
        return failures

    _expect_contains(
        model_recipe,
        [
            'model_type: "openpi"',
            'config_name: "pi0_aloha_robotwin"',
            "num_action_chunks: 50",
            "action_dim: 14",
            'lora_style: "robotwin_pi0_dual_expert"',
            "paligemma_lora_rank: 16",
            "action_expert_lora_rank: 32",
        ],
        failures,
    )
    _expect_contains(
        train_recipe,
        [
            "micro_batch_size: 1",
            "global_batch_size: 32",
            "max_steps: 30000",
            'lr_scheduler: "openpi_cosine"',
            "lr_warmup_steps: 1000",
            "min_lr: 2.5e-6",
            "robotwin/adjust_bottle/norm_stats.json",
        ],
        failures,
    )

    metadata = dataset / "meta/info.json"
    if not metadata.is_file():
        failures.append(f"missing LeRobot metadata: {metadata}")
    stats_path = assets / "norm_stats.json"
    if not stats_path.is_file():
        failures.append(f"missing task normalization statistics: {stats_path}")
    else:
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            fields = stats["norm_stats"]
            for field in ("state", "actions"):
                if field not in fields:
                    failures.append(f"{stats_path}: norm_stats lacks {field!r}")
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            failures.append(f"invalid normalization statistics {stats_path}: {error}")
    return failures


def expected_openpi_lr(
    step: int, peak_lr: float, warmup_steps: int, decay_steps: int, decay_lr: float
) -> float:
    """Pure-Python equivalent of Optax's warmup_cosine_decay_schedule."""
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / (decay_steps - warmup_steps))
    return decay_lr + (peak_lr - decay_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def verify_scheduler(
    peak_lr: float,
    warmup_steps: int,
    decay_steps: int,
    decay_lr: float,
    compare_optax: bool,
) -> list[str]:
    """Compare RLinf's actual LambdaLR values with the JAX/OpenPI definition."""
    if decay_steps <= warmup_steps:
        return ["decay_steps must be greater than warmup_steps"]

    sys.path.insert(0, str(REPO_ROOT))
    import torch

    from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=peak_lr)
    scheduler = get_lr_scheduler(
        "openpi_cosine",
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=decay_steps,
        min_lr=decay_lr,
    )
    steps = [0, 1, warmup_steps - 1, warmup_steps, decay_steps - 1, decay_steps]
    checks = []
    for step in steps:
        torch_lr = scheduler.base_lrs[0] * scheduler.lr_lambdas[0](step)
        expected_lr = expected_openpi_lr(
            step, peak_lr, warmup_steps, decay_steps, decay_lr
        )
        checks.append((step, torch_lr, expected_lr))

    failures = [
        f"step {step}: torch={torch_lr:.12g}, expected={expected_lr:.12g}"
        for step, torch_lr, expected_lr in checks
        if not math.isclose(torch_lr, expected_lr, rel_tol=0.0, abs_tol=1e-15)
    ]
    if compare_optax:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        try:
            import optax
        except ImportError as error:
            return failures + [f"cannot import Optax for --compare-optax: {error}"]
        jax_schedule = optax.warmup_cosine_decay_schedule(
            init_value=peak_lr / (warmup_steps + 1),
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=decay_lr,
        )
        for step, torch_lr, _ in checks:
            jax_lr = float(jax_schedule(step))
            if not math.isclose(torch_lr, jax_lr, rel_tol=1e-6, abs_tol=1e-12):
                failures.append(
                    f"step {step}: torch={torch_lr:.12g}, optax={jax_lr:.12g}"
                )

    print("step       RLinf LR          OpenPI expected LR")
    for step, torch_lr, expected_lr in checks:
        print(f"{step:5d}  {torch_lr:.12g}  {expected_lr:.12g}")
    return failures



def verify_trainable_parameters() -> list[str]:
    """Build the real model and audit the actual pre-FSDP LoRA parameters."""
    os.environ.setdefault("EMBODIED_PATH", str(REPO_ROOT / "examples/sft"))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from hydra import compose, initialize_config_dir
        from rlinf.models import collect_robotwin_pi0_lora_audit, get_model

        with initialize_config_dir(
            version_base=None,
            config_dir=str(REPO_ROOT / "examples/sft/config"),
        ):
            cfg = compose(config_name="robotwin_pi0_aloha_lora")
        model = get_model(cfg.actor.model)
        audit = collect_robotwin_pi0_lora_audit(
            model,
            paligemma_rank=int(cfg.actor.model.openpi.paligemma_lora_rank),
            action_rank=int(cfg.actor.model.openpi.action_expert_lora_rank),
        )
    except Exception as error:  # surfaced as a verification failure, not a traceback
        return [f"could not build/audit the actual LoRA model: {error}"]

    print("RoboTwin pi0 LoRA parameter audit")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return list(audit["errors"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--peak-lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--decay-steps", type=int, default=30000)
    parser.add_argument("--decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--compare-optax", action="store_true")
    parser.add_argument(
        "--check-trainable-parameters",
        action="store_true",
        help="build the real model and verify LoRA targets, ranks, freeze boundary, and init",
    )
    args = parser.parse_args()

    failures = check_recipe(args.dataset, args.assets)
    if args.check_trainable_parameters:
        failures.extend(verify_trainable_parameters())
    failures.extend(
        verify_scheduler(
            args.peak_lr,
            args.warmup_steps,
            args.decay_steps,
            args.decay_lr,
            args.compare_optax,
        )
    )
    if failures:
        print("FAILED", file=sys.stderr)
        print("\n".join(f"- {failure}" for failure in failures), file=sys.stderr)
        return 1
    print("PASS: RoboTwin pi0 ALOHA LoRA recipe and OpenPI schedule are aligned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
