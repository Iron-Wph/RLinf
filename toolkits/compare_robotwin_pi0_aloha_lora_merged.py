#!/usr/bin/env python3
"""Numerically compare a RoboTwin pi0 ALOHA LoRA checkpoint and its merge.

The probe follows RLinf's normal ``predict_action_batch`` preprocessing and
diffusion sampling path. It uses a deterministic ALOHA-shaped observation and
resets Torch/CUDA RNG immediately before each model call, so the two policies
receive identical diffusion noise. It never modifies either checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.models import get_model
from rlinf.scheduler import Worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-checkpoint", required=True, type=Path)
    parser.add_argument("--merged-checkpoint", required=True, type=Path)
    parser.add_argument("--norm-stats", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=20260801, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    return parser.parse_args()


def _model_cfg(checkpoint: Path, norm_stats: Path, *, is_lora: bool):
    """Build exactly the pi0 ALOHA model shape used by the SFT recipe."""
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": str(checkpoint.resolve()),
            "precision": None,
            "load_to_device": True,
            "is_lora": is_lora,
            "lora_rank": 16,
            "num_action_chunks": 50,
            "action_dim": 14,
            "use_proprio": True,
            "num_steps": 10,
            "add_value_head": False,
            "openpi_data": {"norm_stats_path": str(norm_stats.resolve())},
            "openpi": {
                "config_name": "pi0_aloha_robotwin",
                "num_images_in_input": 3,
                "action_chunk": 50,
                "action_env_dim": 14,
                "lora_style": "robotwin_pi0_dual_expert" if is_lora else "",
                "paligemma_lora_rank": 16,
                "action_expert_lora_rank": 32,
                "lora_init_std": 0.01,
                "verify_lora_layout": is_lora,
                "detach_critic_input": True,
                "train_expert_only": False,
            },
        }
    )


def _probe_observation(image_size: int) -> dict[str, Any]:
    """Return one deterministic observation in the RobotwinEnv output format."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    pixels = torch.arange(image_size * image_size * 3, dtype=torch.int32)
    pixels = (pixels.remainder(251) + 2).to(torch.uint8)
    main = pixels.reshape(1, image_size, image_size, 3)
    wrist_views = [
        (main[0].to(torch.int16) + offset).remainder(256).to(torch.uint8)
        for offset in (31, 97)
    ]
    wrist = torch.stack(
        wrist_views,
        dim=0,
    ).unsqueeze(0)
    # Gripper positions remain within the physical ALOHA linear range consumed
    # by the official AlohaInputs transform.
    state = torch.tensor(
        [[0.12, -0.18, 0.21, -0.08, 0.15, -0.11, 0.040,
          -0.13, 0.16, -0.19, 0.09, -0.14, 0.17, 0.041]],
        dtype=torch.float32,
    )
    return {
        "main_images": main,
        "wrist_images": wrist,
        "extra_view_images": None,
        "states": state,
        "task_descriptions": ["adjust the bottle"],
    }


def _predict(
    checkpoint: Path,
    norm_stats: Path,
    observation: dict[str, Any],
    *,
    is_lora: bool,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Seed construction too: this keeps the deliberately unused merged action
    # lm_head stable, although all tensors used by the policy are checkpointed.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    previous_platform = Worker.torch_platform
    previous_device_type = Worker.torch_device_type
    if device.type == "cuda":
        # The production worker moves the base model to CUDA before injecting
        # LoRA. Mirror that order so the initialization/layout audit is run in
        # its production dtype/device, rather than CPU bf16 initialization.
        Worker.torch_platform = torch.cuda
        Worker.torch_device_type = "cuda"
    try:
        model = get_model(_model_cfg(checkpoint, norm_stats, is_lora=is_lora))
    finally:
        Worker.torch_platform = previous_platform
        Worker.torch_device_type = previous_device_type
    model = model.to(device).eval()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        actions, result = model.predict_action_batch(
            observation, mode="eval", compute_values=False
        )
        model_actions = result["forward_inputs"]["model_action"]
        actions = actions.float().cpu().clone()
        model_actions = model_actions.float().cpu().clone()
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return actions, model_actions


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    delta = candidate - reference
    return {
        "shape": list(reference.shape),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "reference_abs_max": float(reference.abs().max().item()),
        "candidate_abs_max": float(candidate.abs().max().item()),
        "exact_equal": bool(torch.equal(reference, candidate)),
    }


def main() -> None:
    args = parse_args()
    for path in (args.lora_checkpoint, args.merged_checkpoint, args.norm_stats):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    observation = _probe_observation(args.image_size)
    lora_actions, lora_model_actions = _predict(
        args.lora_checkpoint, args.norm_stats, observation,
        is_lora=True, device=device, seed=args.seed,
    )
    merged_actions, merged_model_actions = _predict(
        args.merged_checkpoint, args.norm_stats, observation,
        is_lora=False, device=device, seed=args.seed,
    )
    report = {
        "method": "fixed ALOHA-shaped input plus identical Torch/CUDA diffusion RNG",
        "seed": args.seed,
        "image_size": args.image_size,
        "lora_checkpoint": str(args.lora_checkpoint.resolve()),
        "merged_checkpoint": str(args.merged_checkpoint.resolve()),
        "executed_actions": _metrics(lora_actions, merged_actions),
        "pre_output_transform_model_actions": _metrics(
            lora_model_actions, merged_model_actions
        ),
        "note": (
            "The merge accumulates LoRA deltas in fp32, then stores the result in the "
            "backbone bf16 dtype. Exact bit equality is therefore not required; this "
            "report is the numerical verification to inspect alongside environment eval."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()