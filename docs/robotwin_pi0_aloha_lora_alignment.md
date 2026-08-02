# RoboTwin pi0 ALOHA LoRA：结构对齐与行为差分验证

> **历史 PEFT 对照路径。** 当前推荐且已完成 `adjust_bottle` 双卡评测的实现是
> [native JAX-layout LoRA](robotwin_pi0_aloha_native_jax_lora.md)：它在 Pi0
> 模型内部保留 attention-head 维度，而不是使用本文的外层 PEFT adapter。
> 本文保留用于复现实验历史与比较旧 checkpoint；不要将其中的 launcher、eval
> 或 merge 命令用于 native checkpoint。

本文对应 RLinf 的单卡 A800 配方 `robotwin_pi0_aloha_lora`，目标是复现 RoboTwin JAX 配置 `pi0_base_aloha_robotwin_lora` 的**训练接口、LoRA 参数化、冻结边界和学习率数值语义**。它验证的是训练接入和更新数值；RoboTwin 仿真成功率仍须通过独立 rollout 评测，不能由 loss 或学习率曲线自动推出。

> **当前结论（2026-08-01）**：已验证 LoRA 目标模块、rank、初始化、冻结边界、base 权重加载、global batch 和 LR schedule 的结构/数值对齐；但已完成的 30k RLinf checkpoint 在当前 evaluator 中仅为 6/150（4%），而用户提供的官方 JAX LoRA 基准约为 90%。因此它**没有达到 JAX 训练行为或成功率对齐**，文中“对齐”不能解释为成功率对齐。已明确发现训练图像增强的采样粒度和实现不同，详见下文。


## 基准与对齐范围

唯一的学习率真值来自 RoboTwin 的 pi0 JAX 源码：

```text
.robotwin-reference/policy/pi0/src/openpi/training/optimizer.py
CosineDecaySchedule.create()
```

它调用 `optax.warmup_cosine_decay_schedule`，而不是 Transformers/HuggingFace 的 `cosine` scheduler。两者的起点和终点并不等价。

| 项目 | RoboTwin JAX / OpenPI | RLinf pi0 LoRA 配方 |
| --- | --- | --- |
| 模型 | `pi0_base_aloha_robotwin_lora` | `pi0_aloha_robotwin` 底座 + 外层 `robotwin_pi0_dual_expert` LoRA |
| 动作 | 14 维、50-step action chunk | `action_dim: 14`, `num_action_chunks: 50` |
| LoRA | PaliGemma rank 16；action expert rank 32 | `robotwin_pi0_dual_expert`，同样的两个 rank |
| 全局 batch | 32 | 单卡配方：micro batch 16、累积 2 次、global batch 32 |
| 更新步数 | 30000 | `max_steps` / `total_training_steps: 30000` |
| 学习率 | peak `2.5e-5`、warmup 1000、end `2.5e-6` | `openpi_cosine`，同一组参数 |
| 归一化统计 | `norm_stats.actions` 和 `norm_stats.state` | task asset 中的 `adjust_bottle/norm_stats.json` |

`min_lr: 2.5e-6` 对 pi0 很重要。pi0.5 旧示例使用的 `min_lr: 0.0` 不能复用到此 RoboTwin pi0 配方。

## 学习率如何精确对齐

设 peak learning rate 为 `P`，warmup 步数为 `W`，总 decay 步数为 `T`，终点为 `E`。OpenPI 的零基 step `s` 定义为：

```text
init = P / (W + 1)

s < W:
  lr(s) = init + (P - init) * s / W

s >= W:
  progress = min(1, (s - W) / (T - W))
  lr(s) = E + (P - E) * 0.5 * (1 + cos(pi * progress))
```

这里的关键点是：

- `s=0` 从 `P/(W+1)` 开始，而不是从 0 开始；本配方为 `2.4975025e-8`。
- `s=999` 仍处于线性 warmup；`s=1000` 恰好为 peak `2.5e-5`。
- Optax 的 `decay_steps=T` **包含** warmup，所以余弦段长度是 `T-W=29000`，不能把余弦段错误设为 30000 步。
- `s=30000` 为精确终点 `2.5e-6`。30000 次实际参数更新消费的 index 是 0 到 29999；最后一次更新的值和 JAX 一样会略高于终点，随后 scheduler 进入精确终点。这是两端离散 step 语义，不是偏差。

RLinf 的实现位于 `rlinf/hybrid_engines/fsdp/utils.py`：`openpi_cosine` 对每个 optimizer parameter group 构造一个 `LambdaLR`。它从 `initial_lr` 读取 peak，因此恢复训练时不会误把已衰减的当前 lr 当作 peak；每个 group 独立由自身 peak 推导到 `min_lr`。`ref_warmup_cosine` 仍保留为兼容别名，但新配方必须显式使用 `openpi_cosine`。

## 训练接入如何实现

```text
RoboTwin LeRobot 数据集
        + task norm_stats.json
        ↓
OpenPI ALOHA data config（3 路相机、state、action、prompt）
        ↓
pi0 Torch 模型 + 双 expert PEFT LoRA
        ↓
FSDP SFT worker（单卡、梯度累积到 global batch 32）
        ↓
AdamW + openpi_cosine
```

PaliGemma 的视觉塔不在 LoRA target 中；rank-16 adapter 只作用于 PaliGemma 语言部分，rank-32 adapter 作用于 action expert。两个 Gemma 基座被冻结，SigLIP 与 pi0 projection 保持可训练，匹配 RoboTwin 的 JAX freeze-filter 语义。PEFT 默认会冻结传入 PaliGemma 模块的全部参数，因此实现会在 adapter 注入后显式恢复非 `model.language_model.*` 参数的训练状态；否则会错误冻结 SigLIP。

OpenPI JAX 对每个 LoRA 矩阵 A、B 都使用 `normal(stddev=0.01)` 初始化。PEFT 的默认 Gaussian 初始化会把 B 置零，故 RLinf 在注入后显式以 `lora_init_std: 0.01` 重置 A、B；`lora_alpha=rank` 使两边缩放均为 1。单卡 FSDP 的 `use_orig_params: false` 要求每个 flat parameter 的 `requires_grad` 一致，因此会为所有可训练 leaf（LoRA A/B 与 SigLIP）建立独立 FSDP 分组；这只改变 FSDP 分组，LoRA target 仍严格排除 SigLIP，审计会验证两者。

### 为什么不能只用普通 PEFT attention LoRA

官方 JAX 的 `lora.Einsum` 在 q/k/v/o attention projection 中保留了 attention-head 轴；普通 PEFT `Linear` LoRA 会先把 head 展平，再让所有 head 共用一组低秩矩阵。这两种形式即使 rank 相同，参数数和函数族也不同。因此本实现只将 FFN 的 `gate/up/down` 交给 PEFT，并为 q/k/v/o 注入逐头 adapter：

```text
JAX q/k/v:  A[head, in, rank] × B[head, rank, head_dim]
JAX o:      A[head, head_dim, rank] × B[head, rank, out]
RLinf:      相同的张量布局和 einsum，缩放 alpha/rank = 1
```

此外，Torch checkpoint 中 action expert 的通用词表 `lm_head` 不参与 pi0 action 生成，JAX `pi0` action expert 也没有这组参数。本配方在构建模型时将它替换为无参数 `Identity`，使冻结的 action-expert base 与官方实现精确一致；这只改变内存中的训练模型，不会删除或改写任何 checkpoint 文件。

实际由官方 JAX `nnx.eval_shape` 与 RLinf 真实模型审计得到的逐 bucket 数量如下。数值必须一致，验证脚本会在不一致时失败：

| bucket | 官方 JAX | RLinf |
| --- | ---: | ---: |
| PaliGemma LLM base（frozen） | 2,508,531,712 | 2,508,531,712 |
| PaliGemma/SigLIP non-LLM（trainable） | 414,803,696 | 414,803,696 |
| PaliGemma LoRA（trainable） | 27,869,184 | 27,869,184 |
| Action expert base（frozen） | 311,464,960 | 311,464,960 |
| Action expert LoRA（trainable） | 22,118,400 | 22,118,400 |
| pi0 action projection（trainable） | 3,248,160 | 3,248,160 |
| **trainable total** | **468,039,440** | **468,039,440** |

因此这里的“精度对齐”是参数化、初始化、冻结范围、动作/状态归一化与优化器学习率的对齐；它为相同训练行为提供必要条件，但仍不能替代最终 rollout 成功率对比。

使用的实际输入和 asset 路径是：

```text
数据集：/mnt/public2/wph/codes/develop_async/RoboTwin_main_official/data/robotwin/adjust_bottle
统计量：/mnt/public2/wph/models/pi0_base/pi0_base/physical-intelligence/robotwin/adjust_bottle/norm_stats.json
Torch base：/mnt/public2/wph/models/pi0_base/pi0_base
```


## 为什么 RLinf 复用 pi0_aloha_robotwin

当前 PyTorch 接入把两层职责拆开了：pi0_aloha_robotwin 只提供 Pi0/Aloha 底座的数据变换、三路相机、14 维环境 action 到 32 维模型 padding、50-step horizon、prompt 与 norm_stats；外层的 is_lora 和 robotwin_pi0_dual_expert 才向两个 expert 注入 q/k/v/o/gate/up/down 的 LoRA。

官方 JAX 的 pi0_base_aloha_robotwin_lora 则在一个 TrainConfig 中同时定义 LoRA 图和 freeze filter。RLinf 安装的 PyTorch Pi0Config 不会因填入 JAX 的 gemma_2b_lora 或 gemma_300m_lora 名称自动创建 LoRA 层，所以不能仅把 config_name 改成官方 _lora 名；仍必须使用外层注入。复用底座 config 不等于没有 LoRA，但它会掩盖官方完整 recipe 的语义，应该新增一个显式的 pi0_base_aloha_robotwin_lora alias 来固定这层关系。此命名修正本身不会把 4% 提升到 90%。

## 尚未对齐的训练数据过程

已逐行比对官方 JAX preprocess_observation 与安装的 PyTorch preprocess_observation_pytorch。官方 JAX 会为一个 batch 中的**每个样本**分别创建 augmax 随机数并执行 RandomCrop(95%)、Resize、Rotate(-5,5) 和 ColorJitter(0.3,0.4,0.5)。当前 PyTorch 版本则对每个相机的整个 micro-batch 仅采样一组 Torch crop、rotation、brightness、contrast、saturation 随机数，且使用手写 slicing/interpolate/grid_sample 与灰度通道 saturation。

因此，尽管增强范围相近，采样粒度、几何边界和颜色变换分布都不相同。这是已证实的训练输入过程不一致，不能把当前实现称为数据精度对齐；在小演示数据集上它是解释 4% 与 90% 差异的高优先级候选。下一步应将官方 JAX 作为 oracle，对同一固定 batch 比较增强后图像、normalized state/action、noise/time、velocity target、module output 和 gradient。

## 单卡 FSDP 与 LoRA 的兼容设置

本配方在单张 A800 上以一个 rank 运行。PyTorch 会将配置中的 `FULL_SHARD` 自动降级为
`NO_SHARD`；这不是多卡分片训练。此组合下必须使用 `use_orig_params: false`：若设为
`true`，PEFT LoRA 的参数展平/回写会在首次 forward 前报
`Cannot writeback when the parameter shape changes`。这与学习率调度器无关。

已用 `runner.max_steps=1` 完成一次真实数据的 forward、backward 和 optimizer update
验证（记录到 `train/loss`、`train/grad_norm` 和 `train/learning_rate`）。因此完整训练
沿用 YAML 中的 `use_orig_params: false`；请勿为该单卡配方把它改回 `true`。

## 验证接入是否正确

服务器中可用的环境命令是 `switch_env`（不是 `swtich_env`）：

```bash
source /root/.bashrc
source switch_env openpi
cd /mnt/public2/wph/codes/develop_async/RLinf_lora
export JAX_PLATFORMS=cpu
```

先运行一次无需 GPU 训练的结构与 scheduler 验证：

```bash
PYTHONPATH=. python toolkits/verify_robotwin_pi0_aloha_lora.py
```

该脚本会检查：配置中的 pi0/ALOHA/LoRA rank、单卡 batch、OpenPI schedule 名称和端点、LeRobot `meta/info.json`、task `norm_stats.json` 中的 `state/actions` 字段，并打印 `0, 1, 999, 1000, 29999, 30000` 的 RLinf 与 OpenPI 期望 LR。

要验证实际加载后的训练模块和参数，而非仅检查 YAML，请运行：

```bash
PYTHONPATH=. python toolkits/verify_robotwin_pi0_aloha_lora.py \
  --check-trainable-parameters
```

它会构建真实模型并输出 JSON 审计。通过条件是：两个 expert 各有 126 个 LoRA target（18 层 × q/k/v/o/gate/up/down）、PaliGemma rank 16、action expert rank 32；两个 Gemma 基座均为 frozen；SigLIP/non-LLM PaliGemma、pi0 projection 和所有 LoRA A/B 均为 trainable；上述六个 bucket 的参数数与官方 JAX 完全相等；LoRA A/B 的标准差约为 0.01 且 B 非零。任何偏差会使模型构造报错，阻止错误配置开始训练。

要让脚本直接调用安装的 Optax，再增加：

```bash
PYTHONPATH=. python toolkits/verify_robotwin_pi0_aloha_lora.py --compare-optax
```

`JAX_PLATFORMS=cpu` 会使这项纯数值校验在 CPU 上执行。当前安装的 JAX CUDA plugin
版本不兼容时，它仍可能打印 plugin discovery 诊断；只要脚本以 `PASS` 和退出码 0
结束，该诊断不影响 Optax 数值比较或 PyTorch 的 A800 训练。

回归测试命令如下：

```bash
PYTHONPATH=. pytest -q \
  tests/unit_tests/test_openpi_cosine_scheduler.py \
  tests/unit_tests/test_robotwin_pi0_lora_recipe.py
```

其中 `test_openpi_cosine_scheduler.py` 锁定 warmup 起点、warmup 边界、余弦开始、最后训练步、精确 decay 终点，以及多 parameter group 的终点行为。以后若将 `openpi_cosine` 替换成普通 `cosine`，这些测试会立即失败。

最后可做一次真实数据的最小训练 smoke test；它会载入模型和数据，因此会占用 A800，但只执行一次更新：

```bash
bash examples/sft/run_robotwin_pi0_aloha_lora.sh \
  runner.max_steps=1 runner.save_interval=-1
```

日志应包含 `learning_rate`，首个 optimizer update 应使用约 `2.4975e-8`。若进行完整训练，`learning_rate` 指标是在 scheduler step 后记录，因而代表下一次 update 的 LR；将其与上面的零基 step 表对应即可。

在本次实现中，该命令已经以 `CUDA_VISIBLE_DEVICES=1` 在真实 `adjust_bottle` 数据上通过：`loss=0.205`、`grad_norm=1.7`、首步实际 LR=`2.4975025e-8`，退出码为 0。日志中记录的 `learning_rate=5e-8` 是 scheduler 在 step 0 更新后、供 step 1 使用的数值。

## 运行完整 SFT（tmux）

在服务器中启动一个可脱离 SSH 的完整训练：

如果 `ray status` 的默认集群只登记 GPU 0，单纯设置 `CUDA_VISIBLE_DEVICES=1` 不足以改变 Ray 的资源分配。此时不要停止默认 Ray；启动一个仅供本实验使用、同时登记两张物理卡的 head，并通过 `RAY_ADDRESS` 选择它。YAML 的 `cluster.component_placement.actor: 1` 会将训练 worker 固定到物理 GPU 1。

```bash
# 每台机器、每个端口只需要执行一次；不会停止默认 Ray 集群。
cd /mnt/public2/wph/codes/develop_async/RLinf_lora
source /root/.bashrc
source switch_env openpi
env -u CUDA_VISIBLE_DEVICES ray start --head --node-ip-address=127.0.0.1 \
  --port=26379 --dashboard-host=127.0.0.1 --dashboard-port=28265 \
  --num-gpus=2 --temp-dir=/tmp/rlinf_robotwin_cuda1_ray --disable-usage-stats
export RAY_ADDRESS=127.0.0.1:26379
RAY_ADDRESS=$RAY_ADDRESS ray status
```

```bash
tmux new-window -d -t "3:" -n rlinf_lora_train_cuda1 \
  'cd /mnt/public2/wph/codes/develop_async/RLinf_lora && \
   source /root/.bashrc && source switch_env openpi && \
   RAY_ADDRESS=127.0.0.1:26379 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=1 \
   bash examples/sft/run_robotwin_pi0_aloha_lora.sh'
```

查看状态而不进入窗口：

```bash
tmux capture-pane -pt "3:rlinf_lora_train_cuda1" -S -80
tail -f logs/<时间戳>-robotwin_pi0_aloha_lora/run_embodiment.log
```

进入窗口使用 `tmux attach -t 3`，选择 `rlinf_lora_train_cuda1`，按 `Ctrl-b d` 可再次脱离。首步
日志应在一次梯度累积后出现 `train/loss`、`train/grad_norm` 和 `train/learning_rate=5e-8`；后者是
step 0 更新完成后、供 step 1 使用的学习率，step 0 实际使用的学习率为 `2.4975025e-8`。

GPU 选择由两层共同保证：独立 Ray head 以两张物理卡建模，YAML 的 `actor: 1` 让 worker 设置 `CUDA_VISIBLE_DEVICES=1`。启动日志的 PID 必须在 `nvidia-smi` 的 GPU 1 行出现；不要只依据 Ray 的逻辑编号判断物理卡。

## 成功率验证的边界

上述检查只证明统计量、模型结构、LoRA、optimizer 与 LR schedule 的静态接入正确；训练时的图像增强过程已经证实不一致，也没有完成 JAX/PyTorch 前向、梯度和训练轨迹的数值对比。成功率必须按相同场景、instruction protocol、reset seed、rollout RNG 和 episode 集合与 JAX baseline 直接比较。

## 完整 checkpoint 的 RoboTwin 评测与 LoRA 合并

训练完成后的 FSDP 权重在
`checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt`。RLinf 的 OpenPI 加载器支持将对应的
**checkpoint 目录** 直接赋给 `rollout.model.model_path`，并会自动查找该文件。因此评测 LoRA 时
不需要、也不应先合并。不要将 `full_weights.pt` 文件本身赋给 `model_path`，也不要将它填进
`runner.ckpt_path`：后者在模型构造之后用通用路径加载，无法保证 JAX 对齐的 LoRA 结构已经注入。

评测须保留 `is_lora=true`、`lora_style=robotwin_pi0_dual_expert`、PaliGemma rank 16、action
expert rank 32、50 个 action chunks、14 维 ALOHA action、三路图像和 task-specific
`adjust_bottle/norm_stats.json`。仿真资产则应为 `/mnt/public2/wph/models/robotwin_assets`，它与
policy 的 normalization assets 不是同一路径。

`adjust_bottle` 当前有 150 个 eval success seeds。双卡推荐每卡 15 个环境、共 5 个 rollout epoch：

```bash
cd /mnt/public2/wph/codes/develop_async/RLinf_lora
source /root/.bashrc
source switch_env openpi

# 创建独立双卡 Ray head；不要停止其他实验的 Ray。
env -u CUDA_VISIBLE_DEVICES ray start --head --node-ip-address=127.0.0.1 \
  --port=26380 --dashboard-host=127.0.0.1 --dashboard-port=28266 \
  --num-gpus=2 --temp-dir=/tmp/rlinf_robotwin_eval_ray --disable-usage-stats

export RAY_ADDRESS=127.0.0.1:26380
export JAX_PLATFORMS=cpu
export ROBOTWIN_PATH=/mnt/public2/wph/codes/develop_async/RoboTwin
export ROBOT_PLATFORM=ALOHA
CKPT=logs/<train-run>/robotwin_pi0_aloha_lora/checkpoints/global_step_30000
NORM=/mnt/public2/wph/models/pi0_base/pi0_base/physical-intelligence/robotwin/adjust_bottle/norm_stats.json

bash evaluations/run_eval.sh robotwin robotwin_adjust_bottle_openpi_eval \
  'cluster.component_placement={env\, rollout:0-1}' \
  env.eval.assets_path=/mnt/public2/wph/models/robotwin_assets \
  env.eval.total_num_envs=30 env.eval.rollout_epoch=5 \
  env.eval.use_fixed_reset_state_ids=false \
  rollout.model.model_path="$CKPT" rollout.model.is_lora=true \
  rollout.model.lora_rank=16 rollout.model.num_action_chunks=50 \
  rollout.model.action_dim=14 rollout.model.num_steps=10 \
  rollout.model.add_value_head=false \
  rollout.model.openpi.action_chunk=50 rollout.model.openpi.action_env_dim=14 \
  rollout.model.openpi.num_steps=10 rollout.model.openpi.train_expert_only=false \
  +rollout.model.openpi.lora_style=robotwin_pi0_dual_expert \
  +rollout.model.openpi.paligemma_lora_rank=16 \
  +rollout.model.openpi.action_expert_lora_rank=32 \
  +rollout.model.openpi.lora_init_std=0.01 \
  +rollout.model.openpi.verify_lora_layout=true \
  +rollout.model.openpi_data.norm_stats_path="$NORM"
```

最终日志中的 `eval/success_once` 是成功率。为覆盖全部 seed，`use_fixed_reset_state_ids=false`
使每个完成的 episode 选择后续的 success seed；评测结束时应同时记录实际 trajectory 数。

### 合并为无 LoRA 的 RLinf PyTorch checkpoint

本配方不能使用通用 `merge_and_unload()` 或 FSDP converter 的 `merge_lora_weighs`。它们只认识
PEFT MLP adapter，不能完整处理 q/k/v/o 的逐 head adapter，会产生漏合并的错误 policy。下面的专用
脚本拒绝未知或不完整 adapter，且绝不覆盖输入 checkpoint 或已有输出目录：

```bash
CKPT=logs/<train-run>/robotwin_pi0_aloha_lora/checkpoints/global_step_30000
OUT=exports/robotwin_pi0_aloha_lora_global_step_30000_merged

PYTHONPATH=. python toolkits/merge_robotwin_pi0_aloha_lora.py \
  --checkpoint "$CKPT" --output-dir "$OUT"
```

该脚本对每个 attention head 合并 `B[h]^T @ A[h]^T`，对 MLP 合并 `B @ A`。两种更新均以 fp32
累加后转换回 base weight 的 bf16 dtype。完整 export 的 `merge_metadata.json` 必须报告：144 个
attention projection、108 个 MLP projection、504 个移除的 LoRA tensor；权重在
`$OUT/model_state_dict/full_weights.pt`。

合并后沿用上面的环境、seed 和 normalization 配置，只改为：

```bash
rollout.model.model_path="$OUT" \
rollout.model.is_lora=false \
rollout.model.add_value_head=false \
rollout.model.openpi.train_expert_only=false
```

输出是 **RLinf PyTorch** 的普通 state dict，而不是 OpenPI JAX checkpoint；JAX 部署仍需独立进行
PyTorch-to-JAX 命名与分片转换。验证分两层：`test_robotwin_pi0_lora_merge.py` 验证两种 LoRA 的
线性恒等式；随后应使用相同 RoboTwin seeds 分别评测 LoRA 与 merged policy 并比较
`eval/success_once`。
### checkpoint 加载完整性审计

评测模型的构造顺序固定为：先加载 base Pi0，注入 RoboTwin 的双 expert LoRA（或构造普通
non-LoRA 模型），最后读取 FSDP `full_weights.pt`。这是必要的：LoRA checkpoint 的 key 含有
`base_layer`、`adapter.lora_a/lora_b` 与 PEFT `lora_A/lora_B`；若在注入 adapter 之前以
`strict=False` 加载，它们会被静默跳过。加载器现在会输出：

```text
[OPENPI_FULL_CKPT_LOAD] ... is_lora=True tensors=1281 missing=[] unexpected=[]
```

`missing` 或 `unexpected` 中任何非白名单项都会中止评测。非 LoRA merge export 仅允许缺少 action
expert 的 `lm_head.weight`，因为 JAX pi0 action path 不使用且训练配方已将它替换为 `Identity`。

此服务器的仿真 Python package 必须使用
`/mnt/public2/wph/codes/develop_async/RoboTwin`，而不是仅含训练数据的
`RoboTwin_main_official`。同时它要求 `open3d==0.18.0` 与 `mplib==0.2.1`；前者用于通用环境
import，后者提供 `mplib.sapien_utils`。`ASSETS_PATH` 由 RLinf wrapper 从
`env.eval.assets_path=/mnt/public2/wph/models/robotwin_assets` 设置。

### 本次 checkpoint 的双卡评测与 merge 后复测（2026-08-01）

本次使用已完成训练的 `global_step_30000`，在独立双卡 Ray head（`0-1`）、每卡 15 个环境、
5 个 rollout epoch、共 150 条 trajectory 上运行。两次评测的场景、assets、normalization 和
RoboTwin success-seed 轮转配置完全相同：`total_num_envs=30`、`rollout_epoch=5`、
`use_fixed_reset_state_ids=false`、`video_cfg.save_video=false`。

| policy | eval/success_once | 成功数 / trajectory | 完整日志 |
| --- | ---: | ---: | --- |
| 原始 LoRA full checkpoint | 0.04 | 6 / 150 | `logs/20260801-12:39:58-robotwin_adjust_bottle_openpi_eval/eval_embodiment.log` |
| 专用脚本 merged checkpoint | 0.02 | 3 / 150 | `logs/20260801-12:58:25-robotwin_adjust_bottle_openpi_eval/eval_embodiment.log` |

这两个 rollout 的 diffusion 初始噪声并没有在每个 episode 前固定，因此 6/150 与 3/150 的
2 个百分点差异不能单独归因于 merge（两样本比例差的近似标准误约为 1.97 个百分点）。它说明
两份 policy 都能端到端完成 RoboTwin 评测，**不**能作为“merged policy 成功率下降”的结论。
要做成功率显著性比较，应固定/reset 并显式控制 Torch rollout RNG，或扩大到多组独立 seed。

本次 merge 的结构报告在
`exports/robotwin_pi0_aloha_lora_global_step_30000_merged/merge_metadata.json`：
`input_tensor_count=1281`、`output_tensor_count=777`、`merged_attention_projections=144`、
`merged_mlp_projections=108`、`removed_lora_tensors=504`、`lora_scaling=1.0`。这同时证明两个
expert 的全部 18 层 `q/k/v/o/gate/up/down` adapter 都被消费，脚本不会接受部分 merge。

除了仿真成功率，还应执行下面的确定性模型级检查。它使用真实的
`predict_action_batch` 预处理、3 路 224×224 ALOHA 图像、14 维状态、10 次 diffusion step；在原
LoRA 与 merged 模型推理前分别重置相同的 Torch/CUDA RNG。因此两者采样到完全相同的初始 noise：

```bash
cd /mnt/public2/wph/codes/develop_async/RLinf_lora
source /root/.bashrc
source switch_env openpi

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python toolkits/compare_robotwin_pi0_aloha_lora_merged.py \
  --lora-checkpoint logs/20260731-13:59:20-robotwin_pi0_aloha_lora/robotwin_pi0_aloha_lora/checkpoints/global_step_30000 \
  --merged-checkpoint exports/robotwin_pi0_aloha_lora_global_step_30000_merged \
  --norm-stats /mnt/public2/wph/models/pi0_base/pi0_base/physical-intelligence/robotwin/adjust_bottle/norm_stats.json \
  --output logs/validation/lora_merge_action_equivalence_global_step_30000_gpu.json \
  --device cuda:0 --seed 20260801
```

实际结果保存在
`logs/validation/lora_merge_action_equivalence_global_step_30000_gpu.json`：

| 输出 | shape | max abs | mean abs | RMSE |
| --- | --- | ---: | ---: | ---: |
| 反归一化后、交给环境的 actions | `[1, 50, 14]` | 0.00383303 | 0.00069552 | 0.00097653 |
| output transform 前的 model actions | `[1, 1600]` | 0.00877523 | 0.00121136 | 0.00174306 |

两份输出不会 bitwise 相等：运行时 LoRA A/B 是 fp32，而普通 non-LoRA export 必须将
`base + delta` 存回 Pi0 backbone 的 bf16 权重。合并器先以 fp32 计算、最后作一次 bf16 rounding；
上述固定输入与固定 noise 的小数值差正是应检查的量。该检查同时要求加载日志为：LoRA
`missing=[] unexpected=[]`；merged 只允许
`missing=['paligemma_with_expert.gemma_expert.lm_head.weight']`，且 `unexpected=[]`。

对应的回归测试包括：

```bash
PYTHONPATH=. pytest -q \
  tests/unit_tests/test_openpi_cosine_scheduler.py \
  tests/unit_tests/test_robotwin_pi0_lora_recipe.py \
  tests/unit_tests/test_robotwin_pi0_lora_merge.py \
  tests/unit_tests/test_openpi_full_checkpoint_loading.py
```

其中 merge test 用小张量验证 JAX 逐 head `q/k/v/o` 线性恒等式和 PEFT MLP 的 `B @ A`；full
checkpoint loading test 则防止将 FSDP 权重在 LoRA 注入之前以 `strict=False` 静默丢失。
