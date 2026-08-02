"""Regression tests for configurable SFT-to-new-format metadata."""

from __future__ import annotations

import json

import safetensors.torch
import torch

from rlinf.utils.ckpt_convertor.openpi.sft2new import convert


def test_sft2new_copies_an_explicit_architecture_config(tmp_path):
    checkpoint = tmp_path / "global_step_1" / "actor" / "model_state_dict"
    checkpoint.mkdir(parents=True)
    torch.save({"model.weight": torch.ones(2, 3)}, checkpoint / "full_weights.pt")

    input_norm_stats = tmp_path / "input_norm_stats.json"
    input_norm_stats.write_text('{"pi0_aloha_robotwin": {}}', encoding="utf-8")
    config = {
        "action_dim": 32,
        "action_horizon": 50,
        "max_token_len": 48,
        "paligemma_variant": "gemma_2b_lora",
        "action_expert_variant": "gemma_300m_lora",
        "pi05": False,
        "pcd": False,
        "dtype": "bfloat16",
    }
    config_json = tmp_path / "pi0_config.json"
    config_json.write_text(json.dumps(config), encoding="utf-8")

    output_model = tmp_path / "export"
    output_norm_stats = output_model / "physical-intelligence" / "robotwin" / "norm_stats.json"
    convert(
        checkpoint.parents[1],
        input_norm_stats,
        output_model,
        output_norm_stats,
        config_json=config_json,
    )

    assert json.loads((output_model / "config.json").read_text(encoding="utf-8")) == config
    exported = safetensors.torch.load_file(str(output_model / "model.safetensors"))
    assert set(exported) == {"weight"}
    assert exported["weight"].dtype == torch.bfloat16
    assert output_norm_stats.read_text(encoding="utf-8") == input_norm_stats.read_text(encoding="utf-8")
