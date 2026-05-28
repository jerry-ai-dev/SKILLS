# Lesson 1: TRL 库全景 & SFTTrainer 源码

## 学习目标
- 理解 TRL 库的定位和核心模块
- 能读懂 SFTTrainer 的源码架构
- 理解 `DataCollatorForCompletionOnlyLM` 如何实现 Prompt Masking
- 能用 TRL SFTTrainer 跑通最简 SFT

## 🗺️ 今日代码地图

**今天读什么？** 我们读 HuggingFace TRL 库（`pip show trl` 能看到本地路径），它是 SFT/GRPO/DPO 等"后训练"算法的官方实现。

**它干嘛？** 在整个 pipeline 里，TRL 充当"训练器层"——把模型、数据、loss 包装成几行代码就能跑的训练流程。Open-R1 / SimpleRL-Zoo 都是站在 TRL 肩膀上写的。

**主讲文件**：`trl/trainer/sft_trainer.py`（SFTTrainer 类的实现）

**今天涉及的代码清单**：

| 仓库 | 文件 | 关键函数/类 | 在课程中的角色 |
|------|------|-------------|---------------|
| trl | `trl/trainer/sft_trainer.py` | `SFTTrainer` 类 | ⭐ 主文件，今天精读 |
| trl | `trl/trainer/sft_config.py` | `SFTConfig` 类 | 配套：训练超参容器 |
| trl | `trl/trainer/utils.py` | `DataCollatorForCompletionOnlyLM` | M4 精读：Prompt Masking 的承载者 |
| transformers | `transformers/trainer.py` | `Trainer` 父类 | 背景：SFTTrainer 的爹 |
| peft | `peft/mapping.py` | `get_peft_model` | M2 调到：LoRA 包装入口 |

**SFTTrainer 文件内部结构**（4 个主要 section）：

1. **类定义 & 继承** （L100-L150）：`class SFTTrainer(Trainer)` —— 决定能继承谁
2. **`__init__` 初始化逻辑** （L150-L350）：处理 args / 加载模型 / 包装 LoRA / 选 collator / 预处理数据
3. **`_prepare_dataset` 数据预处理** （L400-L550）：应用 chat_template、tokenize、设 -100
4. **训练相关方法**（L600+）：大多数继承自 `Trainer`，本课不细看

## ▶️ 运行体验点（Step 3.4 用）

让学生跑 `lessons/cr_lesson01_runme.py`，**亲眼观察 3 件事**：

1. **看 SFTTrainer 的真实文件位置**：用 `import trl; print(trl.trainer.sft_trainer.__file__)` 找到本地路径，VS Code 打开浏览
2. **看 prompt masking 的真实效果**：用一条简单 SFT 数据走一遍 collator，打印 `batch['labels'][0]`，观察前面那一长串 `-100` 直到 answer 才出现真实 token id
3. **看 LoRA 包装的效果**：调用 `model.print_trainable_parameters()`，对比包 LoRA 前后可训练参数从 5 亿降到几百万

## 知识小节

### 小节 1：TRL 库全景

TRL (Transformer Reinforcement Learning) 是 HuggingFace 提供的后训练一站式工具库。

**核心模块**：

| 模块 | 功能 | 对应阶段二 |
|------|------|-----------|
| `SFTTrainer` | 监督微调训练 | Lesson 7 SFT 理论 |
| `GRPOTrainer` | GRPO 强化学习训练 | Lesson 6 GRPO 算法 |
| `RewardTrainer` | 奖励模型训练 | Lesson 4 Reward Model |
| `DPOTrainer` | 直接偏好优化 | （扩展知识） |
| `DataCollatorForCompletionOnlyLM` | Prompt Masking | Exam 3 Q11 |

**TRL 与 HuggingFace 生态的关系**：
```
transformers (模型 & Tokenizer)
    ↓
datasets (数据加载)
    ↓
peft (LoRA 等高效微调)
    ↓
trl (后训练：SFT + RL)
```

### 小节 2：SFTTrainer 继承体系

SFTTrainer 的继承链：

```python
# trl/trainer/sft_trainer.py（简化版）

class SFTTrainer(transformers.Trainer):
    """SFT 训练器，在 HuggingFace Trainer 基础上增加了：
    1. 自动处理 prompt masking（不计算 prompt 部分的 loss）
    2. 支持 packing（把多条短样本拼成一条长样本，提高 GPU 利用率）
    3. 集成 peft（一行代码开启 LoRA）
    4. 自动应用 chat template
    """
    
    def __init__(
        self,
        model,
        args: SFTConfig,           # 继承自 TrainingArguments，加了 SFT 专用参数
        train_dataset,
        data_collator=None,        # 默认会自动创建 DataCollatorForCompletionOnlyLM
        tokenizer=None,
        peft_config=None,          # 传入 LoRAConfig 自动启用 LoRA
        **kwargs,
    ):
        # 如果传了 peft_config，自动把模型包装成 LoRA 模型
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
        
        # 调用父类 Trainer 的 __init__
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            **kwargs,
        )
```

**SFTTrainer 在 Trainer 基础上加了什么**：
1. `peft_config` 参数 → 一行开启 LoRA
2. 自动数据处理 → 帮你做 chat template + prompt masking
3. `packing` 选项 → 多条短样本拼接，提升效率
4. `SFTConfig` → 继承 `TrainingArguments`，增加了 `max_seq_length`, `packing`, `dataset_text_field` 等参数

### 小节 3：DataCollatorForCompletionOnlyLM — Prompt Masking 的实现

这是阶段二 Exam 3 Q11 的**代码实现版本**。

```python
# trl/trainer/utils.py（核心逻辑简化版）

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """只对 completion（回答）部分计算 loss 的 DataCollator。
    
    原理：
    1. 在 tokenize 后的序列中找到 response_template 的位置
    2. response_template 之前的所有 token 的 label 设为 -100
    3. response_template 之后的 token 保持原样（计算 loss）
    """
    
    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        # response_template 是标记"回答开始"的字符串
        # 例如 "### Response:" 或 "<|assistant|>"
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
    
    def torch_call(self, examples):
        batch = super().torch_call(examples)  # 先调用父类打包成 batch
        
        for i in range(len(batch["labels"])):
            labels = batch["labels"][i]
            input_ids = batch["input_ids"][i]
            
            # 找到 response_template 在 input_ids 中的位置
            response_start = self._find_template_position(
                input_ids, self.response_template_ids
            )
            
            if response_start is not None:
                # response_template 之前的全部设为 -100
                labels[:response_start] = -100
            else:
                # 如果没找到 template，整条样本都不算 loss
                labels[:] = -100
            
            batch["labels"][i] = labels
        
        return batch
    
    def _find_template_position(self, input_ids, template_ids):
        """在 input_ids 中查找 template_ids 子序列的起始位置"""
        template_len = len(template_ids)
        for i in range(len(input_ids) - template_len + 1):
            if input_ids[i:i+template_len].tolist() == template_ids:
                return i + template_len  # 返回 template 之后的位置
        return None
```

**对应阶段二 Exam 3 Q11 的知识**：
- Q11 的 bug 就是没做 prompt masking → TRL 用 `DataCollatorForCompletionOnlyLM` 自动处理
- Q11 手动设 `labels[i, :plen] = -100` → TRL 通过查找 `response_template` 自动定位分界点

### 小节 4：SFTConfig 关键参数

```python
from trl import SFTConfig

config = SFTConfig(
    # === 继承自 TrainingArguments 的常用参数 ===
    output_dir="./output",              # 输出目录
    num_train_epochs=3,                 # 训练轮数
    per_device_train_batch_size=4,      # 每个 GPU 的 batch size（= B）
    gradient_accumulation_steps=4,      # 梯度累积步数（= K）
    learning_rate=2e-5,                 # 学习率
    bf16=True,                          # 使用 bf16 混合精度
    logging_steps=10,                   # 每 10 步打印一次日志
    save_steps=500,                     # 每 500 步保存一次 checkpoint
    
    # === SFT 专用参数 ===
    max_seq_length=2048,                # 最大序列长度
    packing=False,                      # 是否把多条短样本拼成一条
    dataset_text_field="text",          # 数据集中文本字段的名称
)
```

### 小节 5：最简 SFT 完整代码

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# 1. 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# 2. 加载数据集
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# 3. 配置 LoRA
peft_config = LoraConfig(
    r=16,                    # LoRA 秩
    lora_alpha=32,           # LoRA 缩放系数
    target_modules=["q_proj", "v_proj"],  # 哪些层加 LoRA
    lora_dropout=0.05,
)

# 4. 配置训练参数
training_args = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,       # 有效 batch = 4 × 4 = 16
    learning_rate=2e-4,
    bf16=True,
    max_seq_length=512,
    logging_steps=10,
)

# 5. 创建训练器并训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

trainer.train()
```

## 测验题

### Q1（代码阅读，4分）
`DataCollatorForCompletionOnlyLM` 中，如果在 input_ids 里没有找到 response_template，会怎么处理？为什么？

**答案**：整条样本的 labels 全部设为 -100，不计算 loss。因为找不到回答的起始位置，无法区分 prompt 和 answer，与其错误计算不如整条跳过。（4分）

### Q2（概念理解，3分）
SFTTrainer 和 HuggingFace 原生 Trainer 的最大区别是什么？列举 3 个 SFTTrainer 独有的功能。

**答案**：
1. 自动 Prompt Masking（通过 DataCollatorForCompletionOnlyLM）
2. 内置 LoRA 支持（传入 peft_config 自动包装）
3. Packing 功能（多条短样本拼接提升效率）
（每个 1 分）

### Q3（代码补全，3分）
补全以下代码中的 `???` 部分：

```python
from trl import SFTTrainer, SFTConfig
config = SFTConfig(
    output_dir="./output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=???,   # 目标有效 batch size = 64
    bf16=???,                           # 使用 bf16 混合精度
    max_seq_length=???,                 # 最大 1024 个 token
)
```

**答案**：
- `gradient_accumulation_steps=8`（64 / 8 = 8）
- `bf16=True`
- `max_seq_length=1024`
