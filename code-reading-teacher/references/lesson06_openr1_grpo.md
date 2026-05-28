# Lesson 6: Open-R1 GRPO 训练流程

## 学习目标
- 逐行读懂 Open-R1 的 grpo.py 脚本
- 理解 GRPO 与 SFT 脚本的差异
- 掌握 vLLM 推理加速的原理和配置
- 理解关键超参数的调优策略

## 知识小节

### 小节 1：grpo.py 入口脚本

```python
# src/open_r1/grpo.py（核心逻辑简化版）

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from open_r1.rewards import get_reward_funcs

def main():
    # 1. 解析配置
    training_args = GRPOConfig.from_pretrained_or_args(...)
    
    # 2. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad token
    
    # 3. 加载模型（策略模型，参考模型由 GRPOTrainer 自动创建）
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    
    # 4. 加载数据集（只需要 prompt，不需要 answer）
    dataset = load_dataset(training_args.dataset_name)
    
    # 5. 获取奖励函数
    reward_funcs = get_reward_funcs(training_args.reward_funcs)
    
    # 6. 创建 Trainer 并训练
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model()
```

**对比 sft.py 的关键差异**：

| | sft.py | grpo.py |
|--|--------|---------|
| Trainer | SFTTrainer | GRPOTrainer |
| 数据 | prompt + answer | 只需 prompt |
| 模型 | 1 个 | 2 个（策略 + 参考） |
| 额外组件 | 无 | reward_funcs |
| 标签 | 来自数据集 | 由奖励函数动态生成 |

### 小节 2：GRPO 数据集 — 只需 Prompt

```python
# SFT 数据集：需要 prompt + 正确答案
sft_data = {
    "messages": [
        {"role": "user", "content": "3+5=?"},
        {"role": "assistant", "content": "答案是8"},  # 需要提供
    ]
}

# GRPO 数据集：只需要 prompt + 标准答案（用于奖励计算，不是 label）
grpo_data = {
    "prompt": [{"role": "user", "content": "3+5=?"}],
    "answer": "8",  # 只用来算奖励，不是训练标签
}
```

### 小节 3：vLLM 推理加速

GRPO 的每一步都要对每个 prompt 生成 G 条回答。如果 G=16, batch=16，一步就要生成 256 条回答。普通 HuggingFace generate 太慢。

```python
# Open-R1 通过 TRL 集成 vLLM 加速推理
training_args = GRPOConfig(
    use_vllm=True,                    # 启用 vLLM
    vllm_device="cuda:1",             # vLLM 用单独的 GPU
    vllm_gpu_memory_utilization=0.7,  # GPU 显存使用率
)
```

**vLLM 的加速原理**：
1. **PagedAttention**：像操作系统管理内存页一样管理 KV Cache，减少显存碎片
2. **Continuous Batching**：不等所有序列生成完才开始下一批，而是动态调度
3. **推理速度**：比 HF generate 快 5-10 倍

### 小节 4：关键超参数调优

```yaml
# 最关键的 GRPO 超参数

num_generations: 8          # G：每个 prompt 生成几条
# G 太小（2-4）: 组内对比信号弱，方差大
# G 太大（32+）: 显存和计算开销大
# 推荐: 8-16

temperature: 0.7            # 采样温度
# 太低（0.1）: 生成太确定，G 条回答太像
# 太高（1.5+）: 生成太随机，质量差
# 推荐: 0.6-1.0

beta: 0.04                  # KL 惩罚系数
# 太小（0.001）: Reward Hacking 风险
# 太大（1.0+）: 模型不敢动
# 推荐: 0.01-0.1

epsilon: 0.2                # PPO clip 范围
# 标准值 0.1-0.3，通常不需要调

learning_rate: 1.0e-6       # GRPO 的学习率
# 比 SFT 小很多！SFT 一般 2e-5，GRPO 一般 1e-6
# 因为 RL 更不稳定，需要小步更新
```

### 小节 5：GRPO 训练的完整流程

```
SFT 模型（基线）
    │
    ├──→ 复制为参考模型 π_ref（冻结）
    │
    └──→ 策略模型 π_θ（要训练的）
          │
          ↓ 每一步循环：
    ┌─────────────────────────────────┐
    │ 1. 取一批 prompt                  │
    │ 2. π_θ 生成 G 条候选回答           │
    │ 3. 奖励函数打分                    │
    │ 4. 组内 z-score 算优势             │
    │ 5. 算 PPO-Clip loss + KL(π_θ||π_ref)│
    │ 6. 反向传播，更新 π_θ              │
    └─────────────────────────────────┘
```

## 测验题

### Q1（对比分析，4分）
GRPO 数据集和 SFT 数据集的最大区别是什么？为什么 GRPO 不需要提供 assistant 的回答？

**答案**：SFT 数据集需要完整的 prompt+answer（用 answer 做 label 计算 cross_entropy loss），而 GRPO 只需要 prompt（answer 由模型自己生成，由奖励函数打分）。GRPO 的学习信号来自奖励函数，不是来自标注数据。（4分）

### Q2（配置调优，3分）
如果训练中 reward/std 持续为 0，最可能的原因是什么？应该调哪个参数？

**答案**：reward/std=0 说明组内所有回答的奖励相同（全对或全错），没有对比信号。应该增大 num_generations（G）增加多样性，或提高 temperature 让生成更多样。（3分）

### Q3（工程理解，3分）
为什么 GRPO 的学习率（1e-6）比 SFT（2e-5）小 20 倍？

**答案**：RL 训练比 SFT 不稳定得多——策略更新会影响后续采样的分布，形成反馈循环，学习率太大容易导致策略崩溃。SFT 是稳定的监督学习，可以用更大的学习率。（3分）
