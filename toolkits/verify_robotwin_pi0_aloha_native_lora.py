#!/usr/bin/env python3
"""Verify the native JAX-layout RoboTwin Pi0 LoRA construction.

This deliberately checks the model before FSDP wrapping. It is a structural
oracle: base weights must load exactly from the converted JAX-aligned
checkpoint, while the only newly initialized leaves are native LoRA A/B.
"""

from __future__ import annotations

import argparse
import json

import safetensors.torch
import torch

from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

EXPECTED_TRAINABLE = 468_039_440
EXPECTED_LORA_NUMEL = 49_987_584
EXPECTED_LORA_TENSORS = 432


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/mnt/public2/wph/models/pi0_base_pytorch_new",
        help="Directory containing converted model.safetensors.",
    )
    args = parser.parse_args()

    model = Pi0Config(
        pi05=False,
        action_horizon=50,
        action_dim=32,
        max_token_len=48,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        dtype="bfloat16",
        pcd=False,
    ).create()
    state = safetensors.torch.load_file(f"{args.model_path}/model.safetensors", device="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    bad_missing = [name for name in incompatible.missing_keys if "lora" not in name]
    if incompatible.unexpected_keys or bad_missing:
        raise SystemExit(
            "Base loading mismatch: "
            f"unexpected={incompatible.unexpected_keys}, bad_missing={bad_missing}"
        )

    buckets = {"llm_base": [0, 0], "llm_lora": [0, 0], "non_llm": [0, 0]}
    lora_total = lora_nonzero = lora_tensors = lora_nonzero_tensors = 0
    for name, parameter in model.named_parameters():
        is_lora = "lora" in name
        is_llm = name.startswith("llm.")
        trainable = is_lora or not is_llm
        parameter.requires_grad_(trainable)
        bucket = "llm_lora" if is_lora else "llm_base" if is_llm else "non_llm"
        buckets[bucket][0 if trainable else 1] += parameter.numel()
        if is_lora:
            nonzero = int(torch.count_nonzero(parameter.detach()).item())
            lora_total += parameter.numel()
            lora_nonzero += nonzero
            lora_tensors += 1
            lora_nonzero_tensors += int(nonzero > 0)

    trainable = sum(values[0] for values in buckets.values())
    report = {
        "base_state_dict_keys": len(state),
        "missing_lora_tensors": len(incompatible.missing_keys),
        "unexpected_base_tensors": list(incompatible.unexpected_keys),
        "bad_missing_tensors": bad_missing,
        "buckets_trainable_frozen": buckets,
        "trainable_numel": trainable,
        "expected_jax_trainable_numel": EXPECTED_TRAINABLE,
        "lora_total_numel": lora_total,
        "expected_jax_lora_numel": EXPECTED_LORA_NUMEL,
        "lora_tensors": lora_tensors,
        "lora_nonzero_tensors": lora_nonzero_tensors,
        "lora_nonzero_numel": lora_nonzero,
    }
    print("ROBOTWIN_PI0_NATIVE_LORA_VERIFY " + json.dumps(report, sort_keys=True))

    if (
        trainable != EXPECTED_TRAINABLE
        or lora_total != EXPECTED_LORA_NUMEL
        or lora_tensors != EXPECTED_LORA_TENSORS
        or lora_nonzero_tensors != lora_tensors
    ):
        raise SystemExit("Native LoRA audit did not match the JAX Pi0 reference.")


if __name__ == "__main__":
    main()
