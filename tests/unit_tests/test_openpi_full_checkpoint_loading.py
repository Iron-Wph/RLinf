"""Tests for deferred loading of full OpenPI FSDP checkpoints."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.openpi import load_rlinf_full_weights


def test_full_weights_load_after_model_construction(tmp_path):
    source = nn.Linear(3, 2, bias=False)
    target = nn.Linear(3, 2, bias=False)
    weights_dir = tmp_path / "actor" / "model_state_dict"
    weights_dir.mkdir(parents=True)
    torch.save(source.state_dict(), weights_dir / "full_weights.pt")

    assert load_rlinf_full_weights(target, tmp_path, is_lora=True)
    torch.testing.assert_close(target.weight, source.weight)


def test_full_weights_reject_unexpected_or_required_missing_keys(tmp_path):
    target = nn.Linear(3, 2, bias=False)
    weights_dir = tmp_path / "model_state_dict"
    weights_dir.mkdir(parents=True)
    torch.save({"wrong.weight": torch.randn(2, 3)}, weights_dir / "full_weights.pt")

    with pytest.raises(RuntimeError, match="layout mismatch"):
        load_rlinf_full_weights(target, tmp_path, is_lora=True)