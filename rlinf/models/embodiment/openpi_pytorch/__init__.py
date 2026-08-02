# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from typing import Any

import torch

from rlinf.config import torch_dtype_from_precision
from rlinf.utils.logging import get_logger

logger = get_logger()


_ROBOTWIN_NATIVE_LORA_STYLE = "robotwin_pi0_native_jax"
_ROBOTWIN_NATIVE_LORA_EXPECTED_TRAINABLE_NUMEL = 468_039_440


def _is_robotwin_native_lora(cfg: Any, model_cfg: Any) -> bool:
    """Return whether the model must use native JAX-layout Pi0 LoRA."""
    return bool(getattr(cfg, "is_lora", False)) and (
        str(getattr(model_cfg, "lora_style", "")) == _ROBOTWIN_NATIVE_LORA_STYLE
    )


def _configure_robotwin_native_lora(model: Any) -> dict[str, Any]:
    """Apply OpenPI JAX's freeze filter and return a strict layout audit.

    OpenPI JAX freezes every base parameter below ``llm``. It leaves the
    vision encoder and Pi0 projections trainable, and enables only native LoRA
    leaves inside the LLM. Keeping this rule here makes it independent of
    PEFT's generic freeze behavior.
    """
    buckets = {
        "llm_base": {"trainable_numel": 0, "frozen_numel": 0},
        "llm_lora": {"trainable_numel": 0, "frozen_numel": 0},
        "non_llm": {"trainable_numel": 0, "frozen_numel": 0},
    }
    lora_nonzero_numel = 0
    lora_total_numel = 0
    lora_nonzero_tensors = 0
    lora_tensor_count = 0

    for name, parameter in model.named_parameters():
        is_lora = "lora" in name
        is_llm = name.startswith("llm.")
        trainable = is_lora or not is_llm
        parameter.requires_grad_(trainable)

        bucket_name = "llm_lora" if is_lora else "llm_base" if is_llm else "non_llm"
        bucket = buckets[bucket_name]
        bucket["trainable_numel" if trainable else "frozen_numel"] += parameter.numel()
        if is_lora:
            lora_total_numel += parameter.numel()
            nonzero_numel = int(torch.count_nonzero(parameter.detach()).item())
            lora_nonzero_numel += nonzero_numel
            lora_nonzero_tensors += int(nonzero_numel > 0)
            lora_tensor_count += 1

    trainable_numel = sum(
        item["trainable_numel"] for item in buckets.values()
    )
    if trainable_numel != _ROBOTWIN_NATIVE_LORA_EXPECTED_TRAINABLE_NUMEL:
        raise ValueError(
            "RoboTwin native LoRA trainable parameter count mismatch: "
            f"got {trainable_numel}, expected JAX "
            f"{_ROBOTWIN_NATIVE_LORA_EXPECTED_TRAINABLE_NUMEL}."
        )
    if lora_nonzero_tensors != lora_tensor_count:
        raise ValueError(
            "RoboTwin native LoRA contains an all-zero parameter tensor; "
            "JAX initializes both A and B with normal(stddev=0.01)."
        )

    return {
        "style": _ROBOTWIN_NATIVE_LORA_STYLE,
        "trainable_numel": trainable_numel,
        "expected_jax_trainable_numel": _ROBOTWIN_NATIVE_LORA_EXPECTED_TRAINABLE_NUMEL,
        "lora_nonzero_numel": lora_nonzero_numel,
        "lora_total_numel": lora_total_numel,
        "lora_nonzero_tensors": lora_nonzero_tensors,
        "lora_tensor_count": lora_tensor_count,
        "buckets": buckets,
    }


def get_model(cfg: Any, torch_dtype: Any = None) -> Any:
    """Build an OpenPI PyTorch Pi0/Pi0.5 model from ``actor.model`` config.

    ``cfg.model_path`` points at a new-format checkpoint containing
    ``model.safetensors``. Model shape comes from YAML; no checkpoint
    ``config.json`` is read. ``cfg.openpi.task`` selects the SFT, eval, or RL
    wrapper around the shared Pi0 core.
    """
    import pathlib

    import safetensors.torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma as pi0_gemma
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_pytorch.utils.model_builders import (
        _build_eval_model,
        _build_rl_model,
        _build_sft_model,
    )

    model_cfg = cfg.openpi
    native_lora = _is_robotwin_native_lora(cfg, model_cfg)
    # Existing Pi0.5 templates predate the explicit switch, so preserve their
    # behavior by default. Pi0 templates set this field to False explicitly.
    pi05 = bool(OmegaConf.select(cfg, "pi05", default=True))
    target_dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )

    model_path = pathlib.Path(cfg.model_path)
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"openpi_pytorch checkpoint not found: {weights_path}")

    pi0_kwargs = {
        "pi05": pi05,
        "action_horizon": int(cfg.num_action_chunks),
        "action_dim": int(model_cfg.model_action_dim),
        "paligemma_variant": str(model_cfg.paligemma_variant),
        "action_expert_variant": str(model_cfg.action_expert_variant),
        "dtype": "bfloat16",
        "pcd": False,
    }
    discrete_state_input = OmegaConf.select(
        model_cfg, "discrete_state_input", default=None
    )
    if discrete_state_input is not None:
        pi0_kwargs["discrete_state_input"] = bool(discrete_state_input)
    max_token_len = OmegaConf.select(model_cfg, "max_token_len", default=None)
    if max_token_len is not None:
        pi0_kwargs["max_token_len"] = int(max_token_len)

    pi0_config = Pi0Config(**pi0_kwargs)
    model = pi0_config.create()
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    if native_lora:
        incompatible = model.load_state_dict(state_dict, strict=False)
        bad_missing = [name for name in incompatible.missing_keys if "lora" not in name]
        if incompatible.unexpected_keys or bad_missing:
            raise ValueError(
                "RoboTwin native LoRA base-checkpoint loading must be exact except "
                "for newly initialized LoRA leaves; got "
                f"unexpected={incompatible.unexpected_keys}, bad_missing={bad_missing}."
            )
        audit = _configure_robotwin_native_lora(model)
        logger.info(
            "ROBOTWIN_PI0_NATIVE_LORA_AUDIT %s",
            json.dumps(audit, sort_keys=True),
        )
    else:
        model.load_state_dict(state_dict, strict=True)
    n_params = sum(param.numel() for param in model.parameters())
    if target_dtype is not None:
        model = model.to(target_dtype)

    num_steps = int(cfg.num_steps)
    action_chunk = int(cfg.num_action_chunks)
    action_env_dim = int(cfg.action_dim)

    task = OmegaConf.select(model_cfg, "task", default=None)
    if task is None:
        raise ValueError(
            "actor.model.openpi.task is required: set it to 'sft', 'rl', or "
            "'eval' to pick the concrete OpenPI PyTorch model variant."
        )
    task = str(task).lower()

    logger.info(
        "openpi_pytorch[%s]: loaded %s (%.2fB params) %s from %s "
        "precision=%s num_steps=%s",
        task,
        pi0_config,
        n_params / 1e9,
        "strict except native LoRA leaves" if native_lora else "strict",
        weights_path,
        cfg.precision,
        num_steps,
    )

    if task == "eval":
        return _build_eval_model(
            cfg,
            model_cfg,
            model,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
        )

    if task == "sft":
        return _build_sft_model(
            model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
        )

    if task == "rl":
        paligemma_width = pi0_gemma.get_config(pi0_config.paligemma_variant).width
        return _build_rl_model(
            cfg,
            model_cfg,
            model,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
            paligemma_width=paligemma_width,
        )

    raise ValueError(
        f"actor.model.openpi.task={task!r} is not supported; "
        "use 'eval', 'sft', or 'rl'."
    )
