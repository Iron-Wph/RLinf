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

import json
import logging
import torch
import torch.nn as nn
from typing import Callable, Optional

from omegaconf import DictConfig

from rlinf.config import EMBODIED_MODEL, SupportedModel, torch_dtype_from_precision
from rlinf.scheduler import Worker

ModelBuilder = Callable[[DictConfig, Optional[object]], object]
_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(
    model_type: str,
    model_builder: ModelBuilder,
    category: str = "embodied",
    force: bool = False,
):
    """Register a model builder for cfg.model_type."""
    if not model_type:
        raise ValueError("model_type must be a non-empty string.")
    if not callable(model_builder):
        raise TypeError("model_builder must be callable.")
    if not force and model_type in _MODEL_REGISTRY:
        raise ValueError(
            f"Model type `{model_type}` is already registered. "
            "Set force=True to override it."
        )
    _MODEL_REGISTRY[model_type] = model_builder
    SupportedModel.register(model_type, force=force)
    if category == "embodied":
        EMBODIED_MODEL.add(SupportedModel(model_type))


def _register_builtin_models():
    def _build_openvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_openvla_oft(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openvla_oft import get_model

        return get_model(cfg, torch_dtype)

    def _build_openpi(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi import get_model

        return get_model(cfg, torch_dtype)

    def _build_openpi_pytorch(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi_pytorch import get_model

        return get_model(cfg, torch_dtype)

    def _build_dexbotic_pi(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dexbotic_pi import get_model

        return get_model(cfg, torch_dtype)

    def _build_dexbotic_dm0(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dexbotic_dm0 import get_model

        return get_model(cfg, torch_dtype)

    def _build_mlp_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.mlp_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_rlt_mlp_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.mlp_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_cnn_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.cnn_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_flow_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.flow_policy import get_model

        return get_model(cfg, torch_dtype)

    def _build_lingbotvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.lingbotvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_abot_m0(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.abot_m0 import get_model

        return get_model(cfg, torch_dtype)

    def _build_starvla(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.starvla import get_model

        return get_model(cfg, torch_dtype)

    def _build_dreamzero(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.dreamzero import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t_n1d6(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_gr00t_n1d7(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.gr00t import get_model

        return get_model(cfg, torch_dtype)

    def _build_openpi_cfg(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.openpi_cfg import get_model

        return get_model(cfg, torch_dtype)

    def _build_recap_value_model(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.value_model.recap import get_model

        return get_model(cfg, torch_dtype)

    def _build_steam_value_model(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.value_model.steam import get_model

        return get_model(cfg, torch_dtype)

    register_model(
        SupportedModel.OPENVLA.value,
        _build_openvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.OPENVLA_OFT.value,
        _build_openvla_oft,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.OPENPI.value,
        _build_openpi,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.OPENPI_PYTORCH.value,
        _build_openpi_pytorch,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DEXBOTIC_PI.value,
        _build_dexbotic_pi,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DEXBOTIC_DM0.value,
        _build_dexbotic_dm0,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.MLP_POLICY.value,
        _build_mlp_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.RLT_MLP_POLICY.value,
        _build_rlt_mlp_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T.value,
        _build_gr00t,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.CNN_POLICY.value,
        _build_cnn_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.FLOW_POLICY.value,
        _build_flow_policy,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.LINGBOTVLA.value,
        _build_lingbotvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.ABOT_M0.value,
        _build_abot_m0,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.STARVLA.value,
        _build_starvla,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.DREAMZERO.value,
        _build_dreamzero,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.CFG_MODEL.value,
        _build_openpi_cfg,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.RECAP_VALUE_MODEL.value,
        _build_recap_value_model,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.STEAM_VALUE_MODEL.value,
        _build_steam_value_model,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T_N1D6.value,
        _build_gr00t_n1d6,
        category="embodied",
        force=True,
    )
    register_model(
        SupportedModel.GR00T_N1D7.value,
        _build_gr00t_n1d7,
        category="embodied",
        force=True,
    )


_register_builtin_models()


def get_model(cfg: DictConfig):
    model_type = str(cfg.model_type)
    model_builder = _MODEL_REGISTRY.get(model_type)
    if model_builder is None:
        return None

    torch_dtype = torch_dtype_from_precision(cfg.precision)
    model = model_builder(cfg, torch_dtype)

    if (
        Worker.torch_platform is not None
        and Worker.torch_platform.is_available()
        and cfg.get("load_to_device", True)
    ):
        model = model.to(Worker.torch_device_type)

    if cfg.is_lora:
        from peft import LoraConfig, PeftModel, get_peft_model

        lora_style = (
            str(cfg.openpi.get("lora_style", ""))
            if hasattr(cfg, "openpi")
            else ""
        )
        if (
            SupportedModel(model_type) == SupportedModel.OPENPI
            and lora_style == "robotwin_pi0_dual_expert"
        ):
            if hasattr(cfg, "lora_path") and cfg.lora_path is not None:
                raise ValueError(
                    "robotwin_pi0_dual_expert uses two PEFT adapters and cannot "
                    "load them from the legacy actor.model.lora_path. Resume from "
                    "an RLinf FSDP checkpoint instead."
                )
            _apply_robotwin_pi0_dual_expert_lora(model, cfg, get_peft_model)
        if not hasattr(cfg, "lora_path") or cfg.lora_path is None:
            if lora_style != "robotwin_pi0_dual_expert":
                lora_config = LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_rank,
                    lora_dropout=0.0,
                    target_modules=[
                        "proj",
                        "qkv",
                        "fc1",
                        "fc2",  # vision
                        "q",
                        "kv",
                        "fc3",
                        "out_proj",  # project
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        "lm_head",  # llm
                    ],
                    init_lora_weights="gaussian",
                )
                if SupportedModel(model_type) in (
                    SupportedModel.OPENPI,
                    SupportedModel.CFG_MODEL,
                ):
                    module_to_lora = model.paligemma_with_expert.paligemma
                    module_to_lora = get_peft_model(module_to_lora, lora_config)
                    tag_vlm_subtree(model, False)
                    tag_vlm_subtree(module_to_lora, True)
                    model.paligemma_with_expert.paligemma = module_to_lora
                else:
                    model = get_peft_model(model, lora_config)
        elif lora_style != "robotwin_pi0_dual_expert":
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

    return model



_ROBOTWIN_PI0_LORA_TARGETS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
_ROBOTWIN_PI0_LORA_ADAPTERS_PER_EXPERT = 18 * len(_ROBOTWIN_PI0_LORA_TARGETS)

class _RobotwinHeadwiseLoRAAdapter(nn.Module):
    """JAX-equivalent LoRA update for one Gemma attention projection."""

    def __init__(
        self,
        *,
        target: str,
        in_features: int,
        out_features: int,
        num_heads: int,
        head_dim: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        if target not in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            raise ValueError(f"Unsupported RoboTwin attention target: {target}")
        if rank <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("RoboTwin head-wise LoRA dimensions must be positive.")

        self.target = target
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = alpha / rank

        if target == "o_proj":
            if in_features != num_heads * head_dim:
                raise ValueError(
                    f"{target}: expected input {num_heads * head_dim}, got {in_features}"
                )
            shape_a = (num_heads, head_dim, rank)
            shape_b = (num_heads, rank, out_features)
        else:
            if out_features != num_heads * head_dim:
                raise ValueError(
                    f"{target}: expected output {num_heads * head_dim}, got {out_features}"
                )
            shape_a = (num_heads, in_features, rank)
            shape_b = (num_heads, rank, head_dim)

        self.lora_a = nn.Parameter(torch.empty(shape_a))
        self.lora_b = nn.Parameter(torch.empty(shape_b))

    @property
    def weight(self):
        """Expose a trainable leaf for RLinf's FSDP LoRA wrap policy."""
        return self.lora_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.lora_a.to(dtype=x.dtype)
        b = self.lora_b.to(dtype=x.dtype)
        if self.target == "o_proj":
            x_heads = x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)
            delta = torch.einsum("...he,her,hrd->...d", x_heads, a, b)
        else:
            delta = torch.einsum("...d,hdr,hre->...he", x, a, b)
            delta = delta.reshape(*x.shape[:-1], self.out_features)
        return delta * self.scaling


class _RobotwinHeadwiseLoRAProjection(nn.Module):
    """Frozen base projection plus an independently FSDP-wrapped LoRA leaf."""

    def __init__(self, base_layer: nn.Module, adapter: _RobotwinHeadwiseLoRAAdapter):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self):
        """Expose the base projection for upstream PI0 dtype checks."""
        return self.base_layer.weight

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.adapter(x)


def _inject_robotwin_headwise_attention_lora(
    expert, *, layer_prefix: str, rank: int, head_dim: int = 256
):
    """Replace q/k/v/o with JAX-head-wise LoRA projections in one expert."""
    expected_per_target = 18
    counts = {target: 0 for target in ("q_proj", "k_proj", "v_proj", "o_proj")}

    for name, child in list(expert.named_modules()):
        if not name.startswith(layer_prefix) or ".self_attn." not in name:
            continue
        target = name.rsplit(".", 1)[-1]
        if target not in counts:
            continue
        if not hasattr(child, "in_features") or not hasattr(child, "out_features"):
            raise TypeError(f"{name} is not a replaceable Linear projection.")
        if isinstance(child, _RobotwinHeadwiseLoRAProjection):
            raise ValueError(f"{name} already has a RoboTwin head-wise LoRA adapter.")

        in_features = int(child.in_features)
        out_features = int(child.out_features)
        num_heads = (
            in_features // head_dim if target == "o_proj" else out_features // head_dim
        )
        adapter = _RobotwinHeadwiseLoRAAdapter(
            target=target,
            in_features=in_features,
            out_features=out_features,
            num_heads=num_heads,
            head_dim=head_dim,
            rank=rank,
            alpha=float(rank),
        )
        parent_path, attribute = name.rsplit(".", 1)
        setattr(
            expert.get_submodule(parent_path),
            attribute,
            _RobotwinHeadwiseLoRAProjection(child, adapter),
        )
        counts[target] += 1

    invalid = {target: count for target, count in counts.items() if count != expected_per_target}
    if invalid:
        raise ValueError(
            "RoboTwin pi0 expected 18 q/k/v/o attention projections per expert, "
            f"got {counts} for prefix {layer_prefix!r}."
        )


def _remove_robotwin_unused_action_lm_head(action_expert):
    """Drop Torch-only action lm_head; JAX pi0's action expert has no such leaf."""
    lm_head = getattr(action_expert, "lm_head", None)
    if lm_head is None:
        return
    if not isinstance(lm_head, nn.Module):
        raise TypeError("Unexpected non-module action-expert lm_head.")
    action_expert.lm_head = nn.Identity()

def _robotwin_lora_modules(module):
    """Return both JAX-head-wise attention and PEFT FFN LoRA modules."""
    result = []
    for name, child in module.named_modules():
        if isinstance(child, _RobotwinHeadwiseLoRAAdapter):
            result.append(
                {
                    "name": name.removesuffix(".adapter"),
                    "rank": child.rank,
                    "lora_a": child.lora_a,
                    "lora_b": child.lora_b,
                }
            )
            continue
        lora_a = getattr(child, "lora_A", None)
        lora_b = getattr(child, "lora_B", None)
        ranks = getattr(child, "r", None)
        if lora_a is None or lora_b is None or ranks is None:
            continue
        if "default" not in lora_a or "default" not in lora_b:
            continue
        result.append(
            {
                "name": name,
                "rank": int(ranks["default"]),
                "lora_a": lora_a["default"].weight,
                "lora_b": lora_b["default"].weight,
            }
        )
    return result


def _robotwin_lora_weight_stats(modules):
    """Calculate small, JSON-safe initialization statistics without copies."""
    import torch

    stats = {}
    for weight_name in ("lora_a", "lora_b"):
        count = 0
        nonzero = 0
        total = 0.0
        squared_total = 0.0
        for item in modules:
            values = item[weight_name].detach().float()
            count += values.numel()
            nonzero += int(torch.count_nonzero(values).item())
            total += float(values.sum().item())
            squared_total += float(values.square().sum().item())
        mean = total / count if count else 0.0
        variance = max(0.0, squared_total / count - mean * mean) if count else 0.0
        stats[weight_name] = {
            "numel": count,
            "nonzero_numel": nonzero,
            "mean": mean,
            "std": variance**0.5,
        }
    return stats


def _restore_robotwin_pi0_non_llm_trainability(paligemma):
    """Restore the JAX freeze-filter boundary after PEFT freezes its wrapper.

    PEFT intentionally freezes every parameter in the module it receives.  The
    RoboTwin JAX recipe freezes only ``PaliGemma.llm``; SigLIP and any
    non-language PaliGemma projection therefore remain trainable.
    """
    for name, parameter in paligemma.named_parameters():
        if not name.startswith("model.language_model.") and "lora_" not in name:
            parameter.requires_grad_(True)


def _initialize_robotwin_pi0_lora(modules, init_std: float):
    """Match OpenPI JAX's normal(stddev=0.01) initialization for A and B."""
    import torch

    for item in modules:
        torch.nn.init.normal_(item["lora_a"], mean=0.0, std=init_std)
        torch.nn.init.normal_(item["lora_b"], mean=0.0, std=init_std)


def _tag_robotwin_trainable_leaves_for_fsdp(module):
    """Separate all trainable leaves from frozen bases for FSDP flattening.

    This controls only FSDP grouping.  It does not alter the LoRA target list:
    SigLIP leaves are included here because they are trainable in the official
    JAX recipe, while LoRA adapters still exist only in the Gemma projections.
    """
    for child in module.modules():
        if any(child.children()):
            continue
        weight = getattr(child, "weight", None)
        if weight is not None and weight.requires_grad:
            setattr(child, "_to_lora", True)


def collect_robotwin_pi0_lora_audit(model, paligemma_rank: int, action_rank: int):
    """Collect and validate the actual pre-FSDP RoboTwin pi0 LoRA layout."""
    try:
        paligemma = model.paligemma_with_expert.paligemma
        action_expert = model.paligemma_with_expert.gemma_expert
    except AttributeError as exc:
        raise TypeError("Expected OpenPI pi0 dual-expert model for LoRA audit.") from exc

    paligemma_modules = _robotwin_lora_modules(paligemma)
    action_modules = _robotwin_lora_modules(action_expert)
    buckets = {
        "paligemma_llm_base": {"trainable_numel": 0, "frozen_numel": 0, "trainable_tensors": 0, "frozen_tensors": 0},
        "paligemma_non_llm": {"trainable_numel": 0, "frozen_numel": 0, "trainable_tensors": 0, "frozen_tensors": 0},
        "paligemma_lora": {"trainable_numel": 0, "frozen_numel": 0, "trainable_tensors": 0, "frozen_tensors": 0},
        "action_expert_base": {"trainable_numel": 0, "frozen_numel": 0, "trainable_tensors": 0, "frozen_tensors": 0},
        "action_expert_lora": {"trainable_numel": 0, "frozen_numel": 0, "trainable_tensors": 0, "frozen_tensors": 0},
        "pi0_outer_or_projection": {"trainable_numel": 0, "frozen_numel": 0, "trainable_tensors": 0, "frozen_tensors": 0},
    }

    def parameter_category(name: str) -> str:
        paligemma_prefix = "paligemma_with_expert.paligemma."
        action_prefix = "paligemma_with_expert.gemma_expert."
        if name.startswith(paligemma_prefix):
            suffix = name.removeprefix(paligemma_prefix)
            if "lora_" in suffix:
                return "paligemma_lora"
            if suffix.startswith("model.language_model."):
                return "paligemma_llm_base"
            return "paligemma_non_llm"
        if name.startswith(action_prefix):
            return "action_expert_lora" if "lora_" in name else "action_expert_base"
        return "pi0_outer_or_projection"

    errors = []
    expected_grad = {
        "paligemma_llm_base": False,
        "paligemma_non_llm": True,
        "paligemma_lora": True,
        "action_expert_base": False,
        "action_expert_lora": True,
        "pi0_outer_or_projection": True,
    }
    mismatch_examples = []
    for name, parameter in model.named_parameters():
        category = parameter_category(name)
        bucket = buckets[category]
        state = "trainable" if parameter.requires_grad else "frozen"
        bucket[f"{state}_numel"] += parameter.numel()
        bucket[f"{state}_tensors"] += 1
        if parameter.requires_grad != expected_grad[category] and len(mismatch_examples) < 8:
            mismatch_examples.append(name)

    if mismatch_examples:
        errors.append("requires_grad mismatch: " + ", ".join(mismatch_examples))
    # Exact parameter counts from the runtime OpenPI/JAX
    # pi0_base_aloha_robotwin_lora audit. This rejects flattened PEFT attention
    # LoRA and the Torch-only action lm_head.
    expected_buckets = {
        "paligemma_llm_base": {"trainable_numel": 0, "frozen_numel": 2_508_531_712},
        "paligemma_non_llm": {"trainable_numel": 414_803_696, "frozen_numel": 0},
        "paligemma_lora": {"trainable_numel": 27_869_184, "frozen_numel": 0},
        "action_expert_base": {"trainable_numel": 0, "frozen_numel": 311_464_960},
        "action_expert_lora": {"trainable_numel": 22_118_400, "frozen_numel": 0},
        "pi0_outer_or_projection": {"trainable_numel": 3_248_160, "frozen_numel": 0},
    }
    for bucket_name, expected in expected_buckets.items():
        actual = buckets[bucket_name]
        for field, expected_numel in expected.items():
            if actual[field] != expected_numel:
                errors.append(
                    f"{bucket_name}.{field}: got {actual[field]}, "
                    f"expected JAX {expected_numel}"
                )

    for label, modules, expected_rank in (
        ("paligemma", paligemma_modules, paligemma_rank),
        ("action_expert", action_modules, action_rank),
    ):
        if len(modules) != _ROBOTWIN_PI0_LORA_ADAPTERS_PER_EXPERT:
            errors.append(
                f"{label}: expected {_ROBOTWIN_PI0_LORA_ADAPTERS_PER_EXPERT} LoRA target modules, got {len(modules)}"
            )
        invalid_targets = [
            item["name"]
            for item in modules
            if item["name"].rsplit(".", 1)[-1] not in _ROBOTWIN_PI0_LORA_TARGETS
            or item["rank"] != expected_rank
        ]
        if invalid_targets:
            errors.append(f"{label}: invalid LoRA targets/ranks: {invalid_targets[:8]}")

    paligemma_stats = _robotwin_lora_weight_stats(paligemma_modules)
    action_stats = _robotwin_lora_weight_stats(action_modules)
    for label, stats in (("paligemma", paligemma_stats), ("action_expert", action_stats)):
        for weight_name, values in stats.items():
            nonzero_ratio = values["nonzero_numel"] / max(1, values["numel"])
            if nonzero_ratio < 0.999 or not 0.005 <= values["std"] <= 0.015:
                errors.append(
                    f"{label} {weight_name}: expected normal(std≈0.01) with >=99.9% nonzero values, got {values}"
                )

    return {
        "status": "PASS" if not errors else "FAIL",
        "expected": {
            "paligemma_rank": paligemma_rank,
            "action_expert_rank": action_rank,
            "target_modules_per_expert": _ROBOTWIN_PI0_LORA_ADAPTERS_PER_EXPERT,
            "target_suffixes": sorted(_ROBOTWIN_PI0_LORA_TARGETS),
            "lora_init_std": 0.01,
        },
        "adapter_modules": {
            "paligemma_count": len(paligemma_modules),
            "action_expert_count": len(action_modules),
            "paligemma_examples": [item["name"] for item in paligemma_modules[:4]],
            "action_expert_examples": [item["name"] for item in action_modules[:4]],
        },
        "parameter_buckets": buckets,
        "initialization": {
            "paligemma": paligemma_stats,
            "action_expert": action_stats,
        },
        "errors": errors,
    }


def _apply_robotwin_pi0_dual_expert_lora(model, cfg: DictConfig, get_peft_model):
    """Apply RoboTwin's official pi0 LoRA layout to both Gemma experts.

    RoboTwin's ``pi0_base_aloha_robotwin_lora`` recipe freezes the base
    PaliGemma and action-expert LLM weights, then trains rank-16 adapters in
    PaliGemma and rank-32 adapters in the action expert.  The vision tower and
    pi0 projections intentionally remain trainable.  Applying PEFT separately
    to the two LLM modules preserves that freeze boundary; applying it only to
    PaliGemma would silently full-finetune the action expert.
    """
    from peft import LoraConfig

    try:
        paligemma = model.paligemma_with_expert.paligemma
        action_expert = model.paligemma_with_expert.gemma_expert
    except AttributeError as exc:
        raise TypeError(
            "robotwin_pi0_dual_expert requires the OpenPI pi0 model with "
            "paligemma_with_expert.{paligemma,gemma_expert}."
        ) from exc

    # Restrict adapters to the Gemma LLMs. In particular, PaliGemma's vision
    # tower has q_proj/k_proj/v_proj modules too, but RoboTwin's JAX recipe
    # does not inject LoRA into SigLIP.
    # PEFT represents separate gate/up/down FFN factors exactly. Attention is
    # installed below by head-wise adapters because flattened Linear LoRA is
    # not equivalent to OpenPI JAX's per-head Einsum LoRA.
    paligemma_targets = (
        r"^model\.language_model\.layers\..*\."
        r"mlp\.(gate_proj|up_proj|down_proj)$"
    )
    action_expert_targets = (
        r"^model\.layers\..*\."
        r"mlp\.(gate_proj|up_proj|down_proj)$"
    )

    paligemma_rank = int(cfg.openpi.get("paligemma_lora_rank", 16))
    action_expert_rank = int(cfg.openpi.get("action_expert_lora_rank", 32))
    if paligemma_rank <= 0 or action_expert_rank <= 0:
        raise ValueError("RoboTwin pi0 LoRA ranks must both be positive.")

    paligemma_config = LoraConfig(
        r=paligemma_rank,
        lora_alpha=paligemma_rank,
        lora_dropout=0.0,
        target_modules=paligemma_targets,
        init_lora_weights="gaussian",
    )
    action_expert_config = LoraConfig(
        r=action_expert_rank,
        lora_alpha=action_expert_rank,
        lora_dropout=0.0,
        target_modules=action_expert_targets,
        init_lora_weights="gaussian",
    )

    # ``get_peft_model`` injects the LoRA layers into the supplied module in
    # place. Do not retain its PeftModel wrapper here: OpenPI's fused Gemma
    # forward directly accesses ``paligemma.language_model.layers`` and
    # ``gemma_expert.model.layers``, which that outer wrapper would change.
    # RLinf checkpoints the complete FSDP model, so a separate adapter wrapper
    # is not needed for saving or resuming this recipe.
    get_peft_model(paligemma, paligemma_config)
    get_peft_model(action_expert, action_expert_config)
    _inject_robotwin_headwise_attention_lora(
        paligemma,
        layer_prefix="model.language_model.layers.",
        rank=paligemma_rank,
    )
    _inject_robotwin_headwise_attention_lora(
        action_expert,
        layer_prefix="model.layers.",
        rank=action_expert_rank,
    )
    _remove_robotwin_unused_action_lm_head(action_expert)

    # PEFT freezes every parameter in each supplied module.  RoboTwin's JAX
    # freeze filter freezes only the two Gemma ``llm`` trees, so restore SigLIP
    # and any non-language PaliGemma projection parameters afterwards.
    _restore_robotwin_pi0_non_llm_trainability(paligemma)

    init_std = float(cfg.openpi.get("lora_init_std", 0.01))
    if init_std <= 0:
        raise ValueError("RoboTwin pi0 LoRA lora_init_std must be positive.")
    _initialize_robotwin_pi0_lora(_robotwin_lora_modules(paligemma), init_std)
    _initialize_robotwin_pi0_lora(_robotwin_lora_modules(action_expert), init_std)

    audit = collect_robotwin_pi0_lora_audit(
        model, paligemma_rank=paligemma_rank, action_rank=action_expert_rank
    )
    logging.getLogger(__name__).info(
        "[ROBOTWIN_PI0_LORA_AUDIT] %s", json.dumps(audit, sort_keys=True)
    )
    if bool(cfg.openpi.get("verify_lora_layout", True)) and audit["errors"]:
        raise ValueError("RoboTwin pi0 LoRA audit failed: " + "; ".join(audit["errors"]))

    # use_orig_params=False requires every FSDP flat parameter to have a
    # uniform requires_grad state.  Mark all trainable leaves for wrapping so
    # SigLIP does not flatten with frozen Gemma bases.  This is an FSDP layout
    # concern only; the audit above enforces that LoRA targets exclude SigLIP.
    tag_vlm_subtree(model, False)
    _tag_robotwin_trainable_leaves_for_fsdp(paligemma)
    _tag_robotwin_trainable_leaves_for_fsdp(action_expert)


def tag_vlm_subtree(model, is_vlm: bool):
    for n, m in model.named_modules():
        setattr(m, "_to_lora", is_vlm)
