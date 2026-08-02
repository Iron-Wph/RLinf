# RoboTwin `adjust_bottle` Pi0 native JAX-LoRA

This is the canonical recipe for `pi0_base_aloha_robotwin_lora`. It trains the
JAX-aligned PyTorch Pi0 implementation and is intentionally separate from the
older `model_type: openpi` PEFT recipe: PEFT flattens attention heads and uses
different adapter leaves, while OpenPI JAX LoRA is head-wise. The legacy
comparison recipe is retained in
[`robotwin_pi0_aloha_lora_alignment.md`](robotwin_pi0_aloha_lora_alignment.md)
and must not be mixed with this checkpoint format.

## Files and inputs

| Item | Path / value |
| --- | --- |
| Training config | `examples/sft/config/robotwin_pi0_aloha_native_jax_lora.yaml` |
| Model template | `examples/sft/config/model/pi0_pytorch_robotwin_lora.yaml` |
| Launcher | `examples/sft/run_robotwin_pi0_aloha_native_jax_lora.sh` |
| Structural verifier | `toolkits/verify_robotwin_pi0_aloha_native_lora.py` |
| Converted JAX-aligned base | `/mnt/public2/wph/models/pi0_base_pytorch_new` |
| Dataset | `/mnt/public2/wph/codes/develop_async/RoboTwin_main_official/data/robotwin/adjust_bottle` |
| Task normalization | `/mnt/public2/wph/models/pi0_base_pytorch_new/physical-intelligence/robotwin/adjust_bottle/norm_stats.json` |

The new-format checkpoint has 632 base tensors. Do **not** replace it with the
legacy `/mnt/public2/wph/models/pi0_base/pi0_base` checkpoint: its state-dict
layout belongs to the older OpenPI/PEFT implementation.

## What is aligned

The common Pi0, transforms, and RoboTwin dataloader are the versions from the
`feature/openpi-pytorch-jax-aligned-sft` implementation. Native LoRA is added
inside that same Pi0 model, before FSDP wrapping:

| JAX leaf | PyTorch native leaf | Rank |
| --- | --- | --- |
| PaliGemma attention Q/K/V | `HeadwiseLoRALinear.lora_a/b[head,input,rank]...[head,rank,head_dim]` | 16 |
| PaliGemma attention O | `HeadwiseLoRALinear.lora_a/b[head,head_dim,rank]...[head,rank,output]` | 16 |
| PaliGemma FFN | `w_gating_lora_*`, `w_linear_lora_*` | 16 |
| Action-expert equivalents | identical JAX layout | 32 |

Both A and B are initialized with `normal(std=0.01)`, and `alpha/rank=1`.
This is different from conventional PEFT, which commonly zero-initializes B.

The freeze filter is also the JAX filter:

- all base parameters below `llm.` are frozen;
- all native leaves containing `lora` are trainable;
- SigLIP and Pi0 state/action projections (everything outside `llm.`) remain
  trainable.

The resulting exact pre-FSDP counts are:

```text
llm base frozen:       2,819,996,672
native LoRA trainable:    49,987,584
non-LLM trainable:       418,051,856
total trainable:         468,039,440
new LoRA tensors:                  432
```

The configuration uses FP32 master parameters and BF16 autocast on a one-rank
FSDP job. This preserves BF16 model computation while avoiding the FSDP1
single-rank nested-SigLIP `LayerNorm` dtype mismatch caused by parameter-only
BF16 casting. Learning rate uses `openpi_cosine`, with 1,000 warmup steps,
`2.5e-5` peak LR, and `2.5e-6` terminal LR. `seed: 42`, micro batch 16, and
global batch 32 are explicit.

## Verify before a full run

```bash
source /root/.bashrc
source switch_env openpi
cd /mnt/public2/wph/codes/develop_async/RLinf_lora

PYTHONPATH=$PWD python toolkits/verify_robotwin_pi0_aloha_native_lora.py
```

Expected final line contains:

```text
ROBOTWIN_PI0_NATIVE_LORA_VERIFY ...
"base_state_dict_keys": 632
"bad_missing_tensors": []
"unexpected_base_tensors": []
"missing_lora_tensors": 432
"lora_total_numel": 49987584
"trainable_numel": 468039440
```

The full training log additionally prints `ROBOTWIN_PI0_NATIVE_LORA_AUDIT`.
It is emitted after loading the base checkpoint and before FSDP wrapping. A
mismatched base tensor, an unexpected tensor, an incorrect trainable count, or
an all-zero LoRA tensor fails model construction instead of silently training.

For an end-to-end test that covers dataset transforms, task-specific norm
stats, FSDP, forward, backward, optimizer, and scheduler:

```bash
CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cpu \
  bash examples/sft/run_robotwin_pi0_aloha_native_jax_lora.sh \
  runner.max_steps=1 runner.save_interval=999999 data.num_workers=0
```

The validated single-step run reports a finite loss, grad norm, and learning
rate. Its first scheduler value is `5e-8`, the JAX-style warmup start for this
30,000-step/1,000-warmup recipe.

## Start the 30k-step run in tmux

Do not force the stale `RAY_ADDRESS=127.0.0.1:26379` on this server; use the
current Ray session's automatic address discovery.

```bash
tmux new-session -d -s rlinf_native_jax_lora \
  'source /root/.bashrc && source switch_env openpi && \
   cd /mnt/public2/wph/codes/develop_async/RLinf_lora && \
   CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cpu \
   bash examples/sft/run_robotwin_pi0_aloha_native_jax_lora.sh'

tmux capture-pane -pt rlinf_native_jax_lora:0 -S -200
```

The worker placement log must include `local_accelerator_rank=1` and
`visible_accelerators=['1']`. Checkpointing is every 1,000 steps. The launcher
prints the exact run-log directory under `logs/`; the same directory contains
`run_embodiment.log`.

## Resume an interrupted run

Resume takes the **`global_step_<N>` directory**, not its `actor` child and not
`full_weights.pt`. It restores both actor and optimizer state and starts the
next update at `N`:

```bash
RUN=logs/<timestamp>-robotwin_pi0_aloha_native_jax_lora/robotwin_pi0_aloha_native_jax_lora
CKPT="$RUN/checkpoints/global_step_15000"

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cpu \
  bash examples/sft/run_robotwin_pi0_aloha_native_jax_lora.sh \
  runner.resume_dir="$CKPT"
```

The last safe checkpoint from the interrupted experiment in this workspace is
`global_step_15000`; training had progressed past that step, but no later full
checkpoint was written.

## Export and evaluate a checkpoint on two GPUs

`sft2new` creates an adapter-preserving, new-format PyTorch export. It does
**not** merge native LoRA into the base; evaluation must therefore retain
`is_lora: true` and `lora_style: robotwin_pi0_native_jax` from the supplied
eval model config. The generic legacy PEFT merge tool must not be used here.

```bash
source /root/.bashrc
source switch_env openpi
cd /mnt/public2/wph/codes/develop_async/RLinf_lora

RUN=logs/<timestamp>-robotwin_pi0_aloha_native_jax_lora/robotwin_pi0_aloha_native_jax_lora
CKPT="$RUN/checkpoints/global_step_15000"
OUT="exports/robotwin_pi0_aloha_native_jax_lora/global_step_15000_pytorch_new"
NORM_IN=/mnt/public2/wph/models/pi0_base_pytorch_new/physical-intelligence/robotwin/adjust_bottle/norm_stats.json
NORM_OUT="$OUT/physical-intelligence/robotwin/norm_stats.json"

python -m rlinf.utils.ckpt_convertor.openpi.convert sft2new \
  --ckpt "$CKPT" \
  --input-norm-stats "$NORM_IN" \
  --output-model "$OUT" \
  --output-norm-stats "$NORM_OUT" \
  --config-json /mnt/public2/wph/models/pi0_base_pytorch_new/config.json
```

The task norm statistics are deliberately copied to
`physical-intelligence/robotwin/norm_stats.json`: the evaluation transform uses
the generic RoboTwin asset id and must consume the same `adjust_bottle` stats
as SFT.

Start (or reuse) a dedicated two-GPU Ray head once. It does not stop another
Ray cluster:

```bash
env -u CUDA_VISIBLE_DEVICES ray start --head --node-ip-address=127.0.0.1 \
  --port=26380 --dashboard-host=127.0.0.1 --dashboard-port=28266 \
  --num-gpus=2 --temp-dir=/tmp/rlinf_robotwin_eval_ray --disable-usage-stats
```

Then run the 150-seed evaluation (15 environments per GPU × 5 epochs):

```bash
export RAY_ADDRESS=127.0.0.1:26380
export JAX_PLATFORMS=cpu
export ROBOTWIN_PATH=/mnt/public2/wph/codes/develop_async/RoboTwin
export ROBOT_PLATFORM=ALOHA

bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_pytorch_native_lora_eval \
  rollout.model.model_path="$OUT"
```

The placement output must show rollout and environment ranks separately bound
to `visible_accelerators=['0']` and `visible_accelerators=['1']`. The final
metric is `eval/success_once`; logs and rollout videos are written below the
new `logs/<timestamp>-robotwin_adjust_bottle_openpi_pytorch_native_lora_eval/`
directory.

The completed step-15000 run used this exact protocol and produced 150
trajectories with `eval/success_once = 0.8666667` (130/150), equal to
`success_at_end`. This is a measured RoboTwin result for that checkpoint, not
a claim of bitwise equality to a separate JAX training process.

## Logs and regression checks

For training, inspect both the launcher and structured run log:

```bash
tail -f logs/robotwin_pi0_aloha_native_jax_lora_30k/launcher.log
tail -f logs/<timestamp>-robotwin_pi0_aloha_native_jax_lora/run_embodiment.log
```

Run the fast regression suite before changing this recipe:

```bash
PYTHONPATH=$PWD pytest -q \
  tests/unit_tests/test_openpi_cosine_scheduler.py \
  tests/unit_tests/test_openpi_full_checkpoint_loading.py \
  tests/unit_tests/test_robotwin_pi0_lora_recipe.py \
  tests/unit_tests/test_openpi_sft2new_config.py
```

The structural check in the previous section verifies the native attention and
MLP leaves directly against the JAX parameter-count oracle; the two-GPU eval
then verifies the complete data/input/output/simulator path.

## Scope of the guarantee

This verifies that the LoRA parameterization, base-weight loading, freeze
boundary, task normalization, scheduler selection, and a complete optimization
step match the JAX-aligned PyTorch implementation. It does not itself prove a
particular RoboTwin rollout success rate or bit-for-bit equality to a separate
JAX process: those require a fixed-batch JAX/PyTorch loss comparison and an
identical simulator evaluation protocol. The structural and one-step checks
are the prerequisite before interpreting a 30k-step checkpoint's success rate.
