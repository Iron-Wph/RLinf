import math

import pytest
import torch

from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

PEAK_LR = 2.5e-5
WARMUP_STEPS = 1_000
TOTAL_STEPS = 30_000
END_LR = 2.5e-6


def _expected_openpi_lr(step: int) -> float:
    """Pure-Python form of Optax warmup_cosine_decay_schedule."""
    init_multiplier = 1.0 / (WARMUP_STEPS + 1)
    if step < WARMUP_STEPS:
        return PEAK_LR * (
            init_multiplier + (1.0 - init_multiplier) * step / WARMUP_STEPS
        )

    progress = min(1.0, (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return END_LR + (PEAK_LR - END_LR) * cosine


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    return get_lr_scheduler(
        "openpi_cosine",
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_STEPS,
        min_lr=END_LR,
    )


@pytest.mark.parametrize(
    "step", [0, 1, WARMUP_STEPS - 1, WARMUP_STEPS, TOTAL_STEPS - 1, TOTAL_STEPS]
)
def test_openpi_cosine_matches_jax_boundary_values(step: int):
    """Match pi0's JAX/Optax schedule at every phase boundary."""
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=PEAK_LR)
    scheduler = _build_scheduler(optimizer)

    actual_lr = scheduler.base_lrs[0] * scheduler.lr_lambdas[0](step)
    assert actual_lr == pytest.approx(_expected_openpi_lr(step), abs=1e-15)


def test_openpi_cosine_applies_the_zero_based_warmup_start():
    """The first optimizer update consumes the non-zero OpenPI initial LR."""
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=PEAK_LR)
    scheduler = _build_scheduler(optimizer)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(_expected_openpi_lr(0))
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(_expected_openpi_lr(1))


def test_openpi_cosine_uses_each_parameter_groups_peak_lr():
    """A shared absolute min_lr must not derive every group from group zero."""
    first = torch.nn.Parameter(torch.ones(()))
    second = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW(
        [{"params": [first], "lr": 1e-4}, {"params": [second], "lr": 1e-5}]
    )
    scheduler = get_lr_scheduler(
        "openpi_cosine",
        optimizer,
        num_warmup_steps=3,
        num_training_steps=10,
        min_lr=1e-5,
    )

    end_lrs = [
        base_lr * lr_lambda(10)
        for base_lr, lr_lambda in zip(scheduler.base_lrs, scheduler.lr_lambdas)
    ]
    assert end_lrs == pytest.approx([1e-5, 1e-5])


def test_openpi_cosine_rejects_an_empty_decay_interval():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=PEAK_LR)

    with pytest.raises(ValueError, match="total_training_steps"):
        get_lr_scheduler(
            "openpi_cosine",
            optimizer,
            num_warmup_steps=10,
            num_training_steps=10,
            min_lr=END_LR,
        )
