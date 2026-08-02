"""Unit tests for the JAX-style RoboTwin pi0 LoRA checkpoint merger."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as functional

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "toolkits" / "merge_robotwin_pi0_aloha_lora.py"
)
_SPEC = importlib.util.spec_from_file_location("robotwin_lora_merger", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MERGER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MERGER)


def _headwise_runtime(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, target: str
) -> torch.Tensor:
    if target == "o_proj":
        x_heads = x.reshape(*x.shape[:-1], a.shape[0], a.shape[1])
        return torch.einsum("...he,her,hrd->...d", x_heads, a, b)
    delta = torch.einsum("...d,hdr,hre->...he", x, a, b)
    return delta.reshape(*x.shape[:-1], -1)


def test_headwise_q_and_o_merge_reproduces_runtime_linear_update():
    torch.manual_seed(7)
    q_base = torch.randn(4, 3)
    q_a = torch.randn(2, 3, 2)
    q_b = torch.randn(2, 2, 2)
    q_x = torch.randn(5, 3)
    q_merged = _MERGER.merge_headwise_attention_weight(q_base, q_a, q_b, target="q_proj")
    torch.testing.assert_close(
        functional.linear(q_x, q_base) + _headwise_runtime(q_x, q_a, q_b, "q_proj"),
        functional.linear(q_x, q_merged),
        atol=1e-6,
        rtol=1e-6,
    )

    o_base = torch.randn(5, 4)
    o_a = torch.randn(2, 2, 2)
    o_b = torch.randn(2, 2, 5)
    o_x = torch.randn(3, 4)
    o_merged = _MERGER.merge_headwise_attention_weight(o_base, o_a, o_b, target="o_proj")
    torch.testing.assert_close(
        functional.linear(o_x, o_base) + _headwise_runtime(o_x, o_a, o_b, "o_proj"),
        functional.linear(o_x, o_merged),
        atol=1e-6,
        rtol=1e-6,
    )


def test_state_dict_merge_unwraps_both_adapter_kinds():
    torch.manual_seed(9)
    attention_prefix = "expert.model.layers.0.self_attn.q_proj"
    mlp_prefix = "expert.model.layers.0.mlp.gate_proj"
    state = {
        f"{attention_prefix}.base_layer.weight": torch.randn(4, 3),
        f"{attention_prefix}.adapter.lora_a": torch.randn(2, 3, 2),
        f"{attention_prefix}.adapter.lora_b": torch.randn(2, 2, 2),
        f"{mlp_prefix}.base_layer.weight": torch.randn(5, 3),
        f"{mlp_prefix}.lora_A.default.weight": torch.randn(2, 3),
        f"{mlp_prefix}.lora_B.default.weight": torch.randn(5, 2),
        "unrelated.weight": torch.randn(2, 2),
    }
    merged, summary = _MERGER.merge_robotwin_pi0_lora_state_dict(
        state, require_complete_recipe=False
    )
    assert summary["merged_attention_projections"] == 1
    assert summary["merged_mlp_projections"] == 1
    assert summary["removed_lora_tensors"] == 4
    assert f"{attention_prefix}.weight" in merged
    assert f"{mlp_prefix}.weight" in merged
    assert all("lora_" not in key and "base_layer" not in key for key in merged)
    torch.testing.assert_close(merged["unrelated.weight"], state["unrelated.weight"])