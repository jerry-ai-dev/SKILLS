# Lesson 5: Open-R1 SFT 训练流程

## 学习目标
- 能逐行读懂 Open-R1 的 sft.py 训练入口
- 理解 SFT 数据集的选择和格式化
- 掌握 LoRA vs 全参数微调的配置
- 理解分布式训练配置（DeepSpeed / FSDP）

## 知识小节

### 小节 1：sft.py 入口脚本

```python
# src/open_r1/sft.py（核心逻辑简化版）

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

def main():
    # 1. 解析配置（从 YAML 或命令行参数）
    training_args = SFTConfig.from_pretrained_or_args(...)
    
    # 2. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_or_path)
    
    # 3. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name_or_path,
        torch_dtype=torch.bfloat16,       # bf16 加载，省显存
        attn_implementation="flash_attention_2",  # Flash Attention 加速
    )
    
    # 4. 加载数据集
    dataset = load_dataset(training_args.dataset_name)
    
    # 5. 数据预处理：应用 chat template
    def format_example(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
    dataset = dataset.map(lambda x: {"text": format_example(x)})
    
    # 6. 创建 Trainer 并训练
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
```

### 小节 2：SFT 数据集选择

Open-R1 使用的 SFT 数据集通常包含高质量的 Chain-of-Thought 推理数据：

```python
# 常用 SFT 数据集
datasets = {
    "Open-R1-Math-220k": "包含 22 万条数学推理数据，有详细推理步骤",
    "NuminaMath-CoT": "数学竞赛题 + CoT 推理过程",
    "MetaMathQA": "数学推理 QA 数据集",
}

# 数据样例
{
    "messages": [
        {"role": "user", "content": "求解方程 2x + 3 = 7"},
        {"role": "assistant", "content": "让我一步步求解：\n2x + 3 = 7\n2x = 7 - 3\n2x = 4\nx = 2\n\n答案是 x = 2"}
    ]
}
```

**SFT 数据的质量决定了模型的"天花板"**（阶段二 Q15 的结论）。

### 小节 3：SFT 配置文件详解

```yaml
# recipes/Qwen2.5-1.5B-Instruct/sft/config.yaml

model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct

# 数据
dataset_name: HuggingFaceH4/OpenR1-Math-220k
dataset_config: default

# SFT 专用
max_seq_length: 4096
packing: true                    # 多条短样本拼接，提升 GPU 利用率

# 训练参数
per_device_train_batch_size: 2
gradient_accumulation_steps: 8   # 有效 batch = 2 × 8 = 16
num_train_epochs: 2
learning_rate: 2.0e-5
lr_scheduler_type: cosine        # 余弦退火学习率
warmup_ratio: 0.1                # 前 10% 步数做 warmup

# 精度与效率
bf16: true
gradient_checkpointing: true
torch_compile: true              # PyTorch 2.0 编译加速

# 保存
output_dir: ./output/sft
save_strategy: "epoch"
```

### 小节 4：LoRA vs 全参数微调

```python
# 全参数微调（默认）
# 所有参数都参与训练，效果最好但显存需求大
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
# 1.5B 模型全参数训练 ≈ 需要 24GB 显存（bf16）

# LoRA 微调
# 只训练少量额外参数，显存友好
from peft import LoraConfig
peft_config = LoraConfig(
    r=64,                    # LoRA 秩（越大能力越强，但参数越多）
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)
# 1.5B 模型 LoRA 训练 ≈ 需要 8GB 显存
```

| | 全参数微调 | LoRA |
|--|-----------|------|
| 效果 | 更好 | 略差，但够用 |
| 显存 | 大（~24GB for 1.5B） | 小（~8GB for 1.5B） |
| 速度 | 慢 | 快 |
| 适用场景 | 有充足 GPU | 资源有限 |

### 小节 5：分布式训练配置

```yaml
# DeepSpeed ZeRO Stage 2 配置
deepspeed:
  zero_optimization:
    stage: 2                    # 参数+梯度分片
  bf16:
    enabled: true

# 启动命令
# accelerate launch --config_file accelerate_config.yaml src/open_r1/sft.py \
#     --config recipes/Qwen2.5-1.5B-Instruct/sft/config.yaml
```

## 测验题

### Q1（代码阅读，4分）
在 sft.py 中，`attn_implementation="flash_attention_2"` 的作用是什么？为什么要用它？

**答案**：Flash Attention 2 是一种高效的注意力计算实现，通过优化 GPU 内存访问模式来加速 Attention 计算并减少显存占用。对于长序列（如 max_seq_length=4096），普通 Attention 的显存是 O(n²)，Flash Attention 降到接近 O(n)。（4分）

### Q2（配置分析，3分）
如果你只有一张 RTX 3090（24GB 显存），要对 Qwen-1.5B 做 SFT，应该选全参数微调还是 LoRA？为什么？

**答案**：应该选 LoRA。Qwen-1.5B 全参数微调需要约 24GB 显存（刚好打满，没有余量给数据和激活值），而 LoRA 只需约 8GB，留有充足余量。（3分）

### Q3（概念理解，3分）
配置中 `packing: true` 是什么意思？对训练效率有什么影响？

**答案**：Packing 将多条短样本拼接成一条长序列（在 max_seq_length 内），避免 padding 浪费。例如 3 条 200 token 的样本可以拼成 1 条 600 token 的序列。这样 GPU 处理的都是有效 token，提升训练效率。（3分）
