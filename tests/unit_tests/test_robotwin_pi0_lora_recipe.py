import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_robotwin_pi0_lora_recipe_matches_official_settings():
    """Keep the single-A800 recipe aligned with RoboTwin's JAX LoRA setup."""
    model_recipe = (
        REPO_ROOT / "examples/sft/config/model/pi0_robotwin_lora.yaml"
    ).read_text(encoding="utf-8")
    train_recipe = (
        REPO_ROOT / "examples/sft/config/robotwin_pi0_aloha_lora.yaml"
    ).read_text(encoding="utf-8")

    assert 'config_name: "pi0_aloha_robotwin"' in model_recipe
    assert "num_action_chunks: 50" in model_recipe
    assert "action_dim: 14" in model_recipe
    assert 'lora_style: "robotwin_pi0_dual_expert"' in model_recipe
    assert "paligemma_lora_rank: 16" in model_recipe
    assert "action_expert_lora_rank: 32" in model_recipe
    assert "lora_init_std: 0.01" in model_recipe
    assert "verify_lora_layout: true" in model_recipe

    assert "actor: 1" in train_recipe
    assert "micro_batch_size: 1" in train_recipe
    assert "global_batch_size: 32" in train_recipe
    assert "max_steps: 30000" in train_recipe
    assert "min_lr: 2.5e-6" in train_recipe
    assert 'lr_scheduler: "openpi_cosine"' in train_recipe
    assert "use_orig_params: false" in train_recipe
    assert "/data/robotwin/adjust_bottle" in train_recipe
    assert "robotwin/adjust_bottle/norm_stats.json" in train_recipe
    assert "param_dtype: null" in train_recipe
    assert "enabled: true" in train_recipe
    assert 'precision: "bf16"' in train_recipe


def test_robotwin_lora_targets_only_gemma_language_modules():
    """Exclude PaliGemma's SigLIP vision tower from RoboTwin LoRA targets."""
    source = (REPO_ROOT / "rlinf/models/__init__.py").read_text(encoding="utf-8")
    paligemma_pattern = re.search(
        r'paligemma_targets = \(\s*r"([^"]+)"\s*r"([^"]+)"', source
    )
    action_expert_pattern = re.search(
        r'action_expert_targets = \(\s*r"([^"]+)"\s*r"([^"]+)"', source
    )

    assert paligemma_pattern is not None
    assert action_expert_pattern is not None
    paligemma_target = "".join(paligemma_pattern.groups())
    action_expert_target = "".join(action_expert_pattern.groups())

    assert re.fullmatch(
        paligemma_target,
        "model.language_model.layers.0.mlp.down_proj",
    )
    assert re.fullmatch(
        action_expert_target,
        "model.layers.0.mlp.down_proj",
    )
    assert not re.fullmatch(
        paligemma_target,
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
    )
    assert "_restore_robotwin_pi0_non_llm_trainability" in source
    assert "_initialize_robotwin_pi0_lora" in source
    assert "collect_robotwin_pi0_lora_audit" in source
    assert "_tag_robotwin_trainable_leaves_for_fsdp" in source
    assert "_RobotwinHeadwiseLoRAAdapter" in source
    assert "_inject_robotwin_headwise_attention_lora" in source
    assert 'torch.einsum("...d,hdr,hre->...he"' in source
    assert "action_expert.lm_head = nn.Identity()" in source


def test_openpi_sft_loader_enables_datasets4_lerobot_compatibility():
    """The provided OpenPI environment has LeRobot 0.1.0 with datasets 4.x."""
    worker = (REPO_ROOT / "rlinf/workers/sft/fsdp_vla_sft_worker.py").read_text(
        encoding="utf-8"
    )
    data_config = (
        REPO_ROOT / "rlinf/models/embodiment/openpi/dataconfig/__init__.py"
    ).read_text(encoding="utf-8")

    assert "ensure_lerobot_datasets_compat()" in worker
    assert "class _Datasets4ColumnCompat" in data_config
    assert 'Version("4.0.0")' in data_config


def test_robotwin_launcher_keeps_auxiliary_jax_on_cpu():
    """PyTorch LoRA does not require the incompatible optional JAX CUDA plugin."""
    launcher = (REPO_ROOT / "examples/sft/run_robotwin_pi0_aloha_lora.sh").read_text(
        encoding="utf-8"
    )

    assert 'JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"' in launcher


def test_vla_launcher_forwards_hydra_overrides():
    """Smoke-test overrides must reach train_vla_sft.py through the recipe."""
    launcher = (REPO_ROOT / "examples/sft/run_vla_sft.sh").read_text(
        encoding="utf-8"
    )

    assert "shift" in launcher
    assert '"${CMD[@]}" "$@"' in launcher
