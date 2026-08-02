#!/usr/bin/env python3
"""Merge an RLinf RoboTwin pi0 ALOHA dual-expert LoRA checkpoint.

The q/k/v/o projections use OpenPI JAX's per-head LoRA parameterization while
only the MLP projections use PEFT. This tool merges both forms into an ordinary,
non-LoRA RLinf OpenPI state dict. It never modifies the source checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
_ATTENTION_BASE_RE = re.compile(
    r"^(?P<prefix>.+\.self_attn\.(?P<target>q_proj|k_proj|v_proj|o_proj))"
    r"\.base_layer\.weight$"
)
_MLP_BASE_RE = re.compile(
    r"^(?P<prefix>.+\.mlp\.(?P<target>gate_proj|up_proj|down_proj))"
    r"\.base_layer\.weight$"
)
_LORA_STATE_MARKERS = (".adapter.lora_", ".lora_A.default.", ".lora_B.default.")


def _require_tensor(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    """Return a checkpoint tensor or fail with its exact key."""
    value = state_dict.get(key)
    if value is None:
        raise KeyError(f"Required LoRA tensor is missing: {key}")
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor for {key}, got {type(value).__name__}")
    return value


def merge_headwise_attention_weight(
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    target: str,
) -> torch.Tensor:
    """Merge one JAX-equivalent per-head q/k/v/o LoRA adapter.

    The runtime computes x @ A[h] @ B[h] per head. Linear stores its weight
    transposed, so each head's weight update is B[h].T @ A[h].T. The recipe has
    alpha == rank, hence the LoRA scale is exactly one.
    """
    if target not in _ATTENTION_TARGETS:
        raise ValueError(f"Unsupported attention target: {target}")
    if base_weight.ndim != 2 or lora_a.ndim != 3 or lora_b.ndim != 3:
        raise ValueError(
            f"{target}: expected base=[out,in], A/B=[heads,...], got "
            f"{tuple(base_weight.shape)}, {tuple(lora_a.shape)}, {tuple(lora_b.shape)}"
        )
    if not base_weight.is_floating_point():
        raise TypeError(f"{target}: base weight must be floating point")

    per_head_delta = torch.matmul(
        lora_b.float().transpose(-1, -2), lora_a.float().transpose(-1, -2)
    )
    if target == "o_proj":
        # [heads, out, head_dim] -> [out, heads * head_dim]
        delta = per_head_delta.permute(1, 0, 2).reshape(base_weight.shape)
    else:
        # [heads, head_dim, in] -> [heads * head_dim, in]
        delta = per_head_delta.reshape(base_weight.shape)
    return (base_weight.float() + delta).to(dtype=base_weight.dtype)


def merge_peft_mlp_weight(
    base_weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor
) -> torch.Tensor:
    """Merge one PEFT MLP update B @ A for this rank-equals-alpha recipe."""
    if base_weight.ndim != 2 or lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError(
            "Expected MLP base=[out,in], A=[rank,in], B=[out,rank], got "
            f"{tuple(base_weight.shape)}, {tuple(lora_a.shape)}, {tuple(lora_b.shape)}"
        )
    if lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError(
            f"Inconsistent PEFT rank: A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
        )
    if tuple(base_weight.shape) != (lora_b.shape[0], lora_a.shape[1]):
        raise ValueError(
            f"MLP update shape does not match base: base={tuple(base_weight.shape)}, "
            f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}"
        )
    return (base_weight.float() + lora_b.float() @ lora_a.float()).to(
        dtype=base_weight.dtype
    )


def merge_robotwin_pi0_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor], *, require_complete_recipe: bool = True
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return an unwrapped state dict and a JSON-safe merge summary."""
    merged: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    attention_count = 0
    mlp_count = 0
    examples: list[str] = []

    for key, value in state_dict.items():
        attention_match = _ATTENTION_BASE_RE.match(key)
        if attention_match:
            prefix = attention_match.group("prefix")
            target = attention_match.group("target")
            a_key = f"{prefix}.adapter.lora_a"
            b_key = f"{prefix}.adapter.lora_b"
            merged_key = f"{prefix}.weight"
            merged[merged_key] = merge_headwise_attention_weight(
                value,
                _require_tensor(state_dict, a_key),
                _require_tensor(state_dict, b_key),
                target=target,
            )
            consumed.update((key, a_key, b_key))
            attention_count += 1
            if len(examples) < 8:
                examples.append(merged_key)
            continue

        mlp_match = _MLP_BASE_RE.match(key)
        if mlp_match:
            prefix = mlp_match.group("prefix")
            a_key = f"{prefix}.lora_A.default.weight"
            b_key = f"{prefix}.lora_B.default.weight"
            merged_key = f"{prefix}.weight"
            merged[merged_key] = merge_peft_mlp_weight(
                value,
                _require_tensor(state_dict, a_key),
                _require_tensor(state_dict, b_key),
            )
            consumed.update((key, a_key, b_key))
            mlp_count += 1
            if len(examples) < 8:
                examples.append(merged_key)
            continue

        if any(marker in key for marker in _LORA_STATE_MARKERS):
            continue
        merged[key] = value

    unconsumed_lora = sorted(
        key
        for key in state_dict
        if any(marker in key for marker in _LORA_STATE_MARKERS) and key not in consumed
    )
    if unconsumed_lora:
        raise ValueError(
            "Found unsupported or orphaned LoRA tensors; refusing a partial merge: "
            + ", ".join(unconsumed_lora[:8])
        )

    expected_attention = 2 * 18 * len(_ATTENTION_TARGETS)
    expected_mlp = 2 * 18 * len(_MLP_TARGETS)
    if require_complete_recipe and (
        attention_count != expected_attention or mlp_count != expected_mlp
    ):
        raise ValueError(
            "This is not a complete RoboTwin pi0 ALOHA dual-expert checkpoint: "
            f"merged attention={attention_count} (expected {expected_attention}), "
            f"MLP={mlp_count} (expected {expected_mlp})."
        )

    summary: dict[str, Any] = {
        "input_tensor_count": len(state_dict),
        "output_tensor_count": len(merged),
        "merged_attention_projections": attention_count,
        "merged_mlp_projections": mlp_count,
        "removed_lora_tensors": len(consumed) - attention_count - mlp_count,
        "lora_scaling": 1.0,
        "accumulation_dtype": "float32",
        "output_dtype": "base weight dtype",
        "merged_key_examples": examples,
    }
    return merged, summary


def _resolve_input_weights(checkpoint: Path) -> Path:
    """Resolve an RLinf checkpoint directory or a direct full-weight file."""
    if checkpoint.is_file():
        return checkpoint
    for candidate in (
        checkpoint / "model_state_dict" / "full_weights.pt",
        checkpoint / "actor" / "model_state_dict" / "full_weights.pt",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find model_state_dict/full_weights.pt or "
        f"actor/model_state_dict/full_weights.pt below {checkpoint}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="RLinf checkpoint directory or full_weights.pt",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for the non-LoRA export",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and merge in memory without output"
    )
    return parser.parse_args()


def main() -> None:
    """Merge and save without source mutation."""
    args = parse_args()
    source_weights = _resolve_input_weights(args.checkpoint.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not args.dry_run:
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}. "
            "Choose a new output directory."
        )

    print(f"Loading source checkpoint: {source_weights}", flush=True)
    source_state = torch.load(source_weights, map_location="cpu", weights_only=True)
    merged_state, summary = merge_robotwin_pi0_lora_state_dict(source_state)
    summary.update(
        {
            "source_checkpoint": str(source_weights),
            "source_size_bytes": source_weights.stat().st_size,
            "format": "RLinf OpenPI non-LoRA state_dict",
            "note": (
                "The action-expert lm_head is intentionally absent because the "
                "JAX-aligned recipe replaces that unused Torch-only head with Identity."
            ),
        }
    )

    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    weights_dir = output_dir / "model_state_dict"
    weights_dir.mkdir(parents=True, exist_ok=False)
    destination = weights_dir / "full_weights.pt"
    temporary = weights_dir / "full_weights.pt.tmp"
    torch.save(merged_state, temporary)
    os.replace(temporary, destination)
    with (output_dir / "merge_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Saved merged checkpoint: {destination}", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise