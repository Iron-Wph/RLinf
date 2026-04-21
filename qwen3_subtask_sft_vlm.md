# Qwen3-VL 子任务监督微调使用文档

本文档介绍 RLinf 中 **Qwen3-VL 子任务监督微调（SFT）** 的实现方式、数据格式、训练配置和启动方法。

适用场景：

- 输入为 **两张图像** + **全局任务描述**
- 输出为当前步骤的 **子任务描述**
- 使用 `qwen3_vl` / `qwen3_vl_moe` 类 VLM 进行监督微调

## 环境与依赖说明：

- 使用 **J02** 镜像：`qwen3vl_instruct_test`
- 请确保 `transformers` 版本正确，建议安装最新主干版本：

```bash
pip install git+https://github.com/huggingface/transformers
# pip install transformers==4.57.0 # currently, V4.57.0 is not released
```

---

## 1. 功能概述

这个任务的目标是让模型根据：

- `task_instruction`：全局任务指令
- `initial_image`：初始状态图像
- `current_image`：当前状态图像

预测当前正在执行的子任务，例如：

- `move the cup to the right`
- `open the drawer`
- `pick up the red block`

在 RLinf 中，这个流程由以下几部分组成：

- `examples/sft/train_vlm_sft.py`：训练入口
- `rlinf/workers/sft/fsdp_vlm_sft_worker.py`：VLM SFT 训练 worker
- `rlinf/data/datasets/vlm.py`：子任务数据集定义
- `examples/sft/config/qwen3_subtask_sft_vlm.yaml`：训练配置
- `examples/sft/run_vlm_sft.sh`：运行脚本

---

## 2. 实现原理

### 2.1 训练入口

`examples/sft/train_vlm_sft.py` 会完成以下工作：

1. 读取 Hydra 配置
2. 通过 `validate_cfg(cfg)` 做配置校验
3. 创建 `Cluster`
4. 构造 `HybridComponentPlacement`
5. 启动 `FSDPVlmSftWorker`
6. 创建 `SFTRunner`
7. 根据 `train_data_paths` 决定是训练还是只评估

如果配置里：

- `data.train_data_paths` 不为空：执行训练
- `data.train_data_paths` 为空：只执行评估

### 2.2 数据集构建

子任务监督使用的是注册式数据集：

- `dataset_name: "robo2vlm_subtask_sft"`
- 对应类：`Robo2VLMSubtaskSFTDataset`

该数据集继承自 `Robo2VLMSFTDataset`，核心逻辑在 `rlinf/data/datasets/vlm.py` 中：

- 读取 `task_instruction` 作为主 prompt
- 读取 `subtask_instruction` 作为监督答案
- 读取 `initial_image` 和 `current_image` 作为多模态输入
- 将 prompt 拼成 VLM chat 格式
- 通过 `AutoProcessor.apply_chat_template` 生成 token 和图像输入
- 生成 `attention_mask` 与 `label_mask`

### 2.3 标签构造方式

这个任务不是做分类，而是做 **自由文本生成**：

- 模型输入：任务 + 两张图 + 系统提示词
- 模型输出：子任务文本

SFT 的监督信号来自 `subtask_instruction`。

在 batch 里：

- `attention_mask` 用于标记有效 token
- `label_mask` 用于标记需要计算 loss 的位置

也就是说，prompt 部分会被 mask，loss 只回传到答案部分。

### 2.4 训练 worker

`FSDPVlmSftWorker` 负责：

- 构建 tokenizer
- 构建 `VLMDatasetRegistry` 中的子任务数据集
- 使用 `sft_collate_fn` 做 batch 拼接
- 处理 checkpoint 保存和恢复
- 在评估阶段提取模型输出并做文本对比

支持的模型类型包括：

- `qwen2.5_vl`
- `qwen3_vl`
- `qwen3_vl_moe`

当前这个子任务配置使用的是：

- `model_type: "qwen3_vl"`

---

## 3. 数据格式要求

建议数据保存为 `parquet`，并至少包含以下字段：

| 字段名 | 含义 | 是否必须 |
|---|---|---|
| `task_instruction` | 全局任务描述 | 是 |
| `subtask_instruction` | 监督目标，子任务文本 | 是 |
| `initial_image` | 初始状态图像 | 是 |
| `current_image` | 当前状态图像 | 是 |

### 3.1 图像字段格式

`image_keys` 支持以下类型：

- `PIL.Image`
- `bytes`
- 图像路径字符串
- URL / URI 字符串

### 3.2 数据集组织建议

推荐将训练集和验证集分开存放：

```text
/path/to/dataset/
  train/
    subtask_sft_train.parquet
  eval/
    subtask_sft_eval.parquet
```

如果你只是想先验证流程，也可以只配置 `eval_data_paths`。

---

## 4. 训练 YAML 配置说明

推荐直接参考：

- `examples/sft/config/qwen3_subtask_sft_vlm.yaml`

下面是关键配置项说明。

### 4.1 runner

```yaml
runner:
  task_type: sft
  max_epochs: 6000
  max_steps: -1
  val_check_interval: 1000
  save_interval: 1000
  experiment_name: qwen3_subtask_sft_vlm
  output_dir: ../results
```

说明：

- `task_type: sft`：必须是 SFT 任务
- `max_epochs`：训练轮数
- `val_check_interval`：每多少 step 做一次评估
- `save_interval`：每多少 step 保存一次 checkpoint
- `experiment_name`：实验名，会出现在日志目录里

### 4.2 data

```yaml
data:
  type: vlm
  dataset_name: "robo2vlm_subtask_sft"
  apply_chat_template: True
  use_chat_template: True
  train_data_paths: /path/to/subtask_sft_train.parquet
  eval_data_paths: /path/to/subtask_sft_eval.parquet
  prompt_key: "task_instruction"
  answer_key: "subtask_instruction"
  image_keys: ["initial_image", "current_image"]
  system_prompt: >-
    You are a robotic assistant specialized in subtask planning...
  max_prompt_length: 1024
  lazy_loading: false
  num_workers: 16
```

说明：

- `type: vlm`：使用 VLM 数据管线
- `dataset_name: "robo2vlm_subtask_sft"`：绑定到子任务数据集
- `train_data_paths`：训练数据路径，不填则只做评估
- `eval_data_paths`：验证数据路径
- `prompt_key`：任务描述字段名
- `answer_key`：监督答案字段名
- `image_keys`：多模态图像字段名，顺序会直接进入 prompt
- `system_prompt`：可选，系统提示词
- `max_prompt_length`：prompt 最大长度
- `lazy_loading`：是否懒加载读取数据

### 4.3 actor

```yaml
actor:
  training_backend: "fsdp"
  micro_batch_size: 2
  eval_batch_size: 8
  global_batch_size: 128
  seed: 42

  model:
    model_type: "qwen3_vl"
    precision: fp32
    model_path: /path/to/Qwen3-VL-4B-Instruct
    is_lora: False
```

说明：

- `training_backend: fsdp`：子任务 SFT 目前走 FSDP 路线
- `model_type: qwen3_vl`：要和 `FSDPVlmSftWorker` 支持的类型一致
- `model_path`：本地模型目录，必须能被 `AutoTokenizer` / `AutoProcessor` 读取
- `micro_batch_size`：单卡微批次
- `global_batch_size`：全局 batch size

### 4.4 其他开关

```yaml
algorithm:
  adv_type: gae

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

说明：

- SFT 任务虽然保留了 `algorithm`、`reward`、`critic` 结构，但这里不启用 RL 相关分支
- `use_reward_model: False`、`use_critic_model: False` 是合理默认值

---

## 5. 配置文件示例

下面给出一个最小可用的子任务 SFT 配置骨架：

```yaml
defaults:
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null

cluster:
  num_nodes: 1
  component_placement:
    actor: all

runner:
  task_type: sft
  logger:
    log_path: ${runner.output_dir}/${runner.experiment_name}
    project_name: rlinf
    experiment_name: ${runner.experiment_name}
    logger_backends: ["tensorboard"]
  max_epochs: 6000
  max_steps: -1
  val_check_interval: 1000
  save_interval: 1000
  experiment_name: qwen3_subtask_sft_vlm
  output_dir: ../results

data:
  type: vlm
  dataset_name: "robo2vlm_subtask_sft"
  apply_chat_template: True
  use_chat_template: True
  train_data_paths: /path/to/subtask_sft_train.parquet
  eval_data_paths: /path/to/subtask_sft_eval.parquet
  prompt_key: "task_instruction"
  answer_key: "subtask_instruction"
  image_keys: ["initial_image", "current_image"]
  system_prompt: >-
    You are a robotic assistant specialized in subtask planning.
  max_prompt_length: 1024
  lazy_loading: false
  num_workers: 16

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 2
  eval_batch_size: 8
  global_batch_size: 128
  seed: 42

  model:
    model_type: "qwen3_vl"
    precision: fp32
    model_path: /path/to/Qwen3-VL-4B-Instruct
    is_lora: False

  optim:
    lr: 1e-5
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    lr_scheduler: "cosine"
    total_training_steps: ${runner.max_epochs}
    lr_warmup_steps: 200

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

---

## 6. 启动方式

### 6.1 使用脚本

脚本默认配置名是：

- `examples/sft/config/qwen3_subtask_sft_vlm.yaml`

如果你在 Bash 环境里运行，可以直接执行：

```bash
bash examples/sft/run_vlm_sft.sh 5
```

上面的 `5` 会被当作任务编号，脚本会自动拼出：

- `task-0005/subtask_sft_eval.parquet`

### 6.2 直接启动训练入口

如果你不想用脚本，也可以直接调用：

```bash
python examples/sft/train_vlm_sft.py \
  --config-path examples/sft/config/ \
  --config-name qwen3_subtask_sft_vlm \
  runner.logger.log_path=../results/qwen3_subtask_sft_vlm
```

如果需要覆盖数据路径：

```bash
python examples/sft/train_vlm_sft.py \
  --config-path examples/sft/config/ \
  --config-name qwen3_subtask_sft_vlm \
  data.train_data_paths=/path/to/subtask_sft_train.parquet \
  data.eval_data_paths=/path/to/subtask_sft_eval.parquet
```

---

## 7. 只做评估

如果你只想跑 eval，不训练，可以把：

```yaml
data:
  train_data_paths: null
```

然后保留 `eval_data_paths`。

此时 `train_vlm_sft.py` 会自动进入 `runner.run_eval()`。

---

## 8. 输出与日志

训练过程中常见的产物包括：

- `logs/<experiment_name>/...`：日志目录
- `checkpoints/global_step_<N>/`：阶段性 checkpoint
- `tensorboard`：训练曲线

如果你启用了评估打印，还会输出部分预测样本，方便快速检查生成质量。

---

## 9. 常见问题

### 9.1 模型路径报错

检查：

- `actor.model.model_path` 是否存在
- 权重目录里是否包含 tokenizer / processor 所需文件

### 9.2 数据字段对不上

检查 YAML 里的：

- `prompt_key`
- `answer_key`
- `image_keys`

是否和 parquet 列名一致。

### 9.3 训练跑不起来或显存不足

优先尝试：

- 降低 `micro_batch_size`
- 降低 `num_workers`
- 缩短 `max_prompt_length`
- 换更小的模型

### 9.4 只想先验证管线

建议：

- 用很小的训练集和验证集
- `max_epochs: 1`
- `save_interval: 1`
- 先确认 prompt、图像和答案都能正常进模型

---

## 10. 关键代码索引

如果你想继续修改或扩展这个任务，建议优先看这几个文件：

- `rlinf/data/datasets/vlm.py`
- `rlinf/data/datasets/__init__.py`
- `rlinf/workers/sft/fsdp_vlm_sft_worker.py`
- `examples/sft/train_vlm_sft.py`
- `examples/sft/config/qwen3_subtask_sft_vlm.yaml`
- `examples/sft/run_vlm_sft.sh`

---

## 11. 一句话总结

Qwen3-VL 子任务 SFT 的本质是：**把“全局任务 + 双图像上下文”包装成 VLM chat 输入，用 `subtask_instruction` 作为监督答案，在 FSDP SFT 流程里做自由文本生成训练。**
