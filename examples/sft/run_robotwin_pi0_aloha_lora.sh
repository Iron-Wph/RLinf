#!/usr/bin/env bash
set -euo pipefail

# One-process recipe for a single A800. Override CUDA_VISIBLE_DEVICES to select
# another device, and edit data.train_data_paths in the YAML (or pass it as a
# Hydra override) before launching.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# OpenPI's PyTorch loader queries JAX process metadata. Keep that auxiliary
# dependency on CPU because this environment's JAX CUDA plugin is optional.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_vla_sft.sh" robotwin_pi0_aloha_lora "$@"
