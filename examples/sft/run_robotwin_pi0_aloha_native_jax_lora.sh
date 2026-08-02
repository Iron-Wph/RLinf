#!/usr/bin/env bash
set -euo pipefail

# One physical A800, GPU 1. The JAX package is only used by the data transforms
# in this PyTorch training process, so keep it off CUDA.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_vla_sft.sh" robotwin_pi0_aloha_native_jax_lora "$@"
