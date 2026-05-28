# Lesson 8: SimpleRL-Zoo 小模型 RL 实验

## 学习目标
- 理解 SimpleRL-Zoo 的定位和设计理念
- 对比 SimpleRL-Zoo、TRL、Open-R1 三者的异同
- 掌握在消费级 GPU 上跑 GRPO 的资源估算
- 动手跑一个最简 GRPO toy 实验

## 知识小节

### 小节 1：SimpleRL-Zoo 是什么

SimpleRL-Zoo（https://github.com/hkust-nlp/simpleRL-reason）专注于小模型（≤3B）的 RL 实验。

**设计理念**：用最简单的代码实现核心 RL 训练，方便研究者在有限资源下做实验。

### 小节 2：与 TRL/Open-R1 的对比

| 维度 | TRL | Open-R1 | SimpleRL-Zoo |
|------|-----|---------|-------------|
| 定位 | 通用后训练工具库 | 复现 DeepSeek R1 | 小模型 RL 实验 |
| 代码量 | 大（数万行） | 中（数千行） | 小（数百行核心） |
| 模型规模 | 任意 | 1.5B-70B | 0.5B-3B |
| GPU 需求 | 灵活 | 多卡 H100 | 单卡 3090 即可 |
| 推理加速 | 可选 vLLM | 集成 vLLM | HF generate |
| 适合谁 | 工程部署 | 大规模复现 | 快速实验验证 |

### 小节 3：SimpleRL-Zoo 核心训练脚本

```python
# SimpleRL-Zoo 的 GRPO 训练核心（极简版）

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def grpo_train_step(model, ref_model, tokenizer, prompts, answers, 
                     G=4, beta=0.1, epsilon=0.2):
    """一步 GRPO 训练"""
    
    # Step 1: 采样 — 每个 prompt 生成 G 条回答
    all_completions = []
    all_log_probs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        for _ in range(G):
            output = model.generate(**inputs, max_new_tokens=256, 
                                     do_sample=True, temperature=0.7)
            completion = tokenizer.decode(output[0], skip_special_tokens=True)
            all_completions.append(completion)
    
    # Step 2: 打分
    rewards = compute_rewards(all_completions, answers, G)
    
    # Step 3: 算优势（z-score）
    rewards = rewards.view(-1, G)  # [batch, G]
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean) / (std + 1e-8)
    
    # Step 4: 算 loss
    # ... (PPO-Clip + KL penalty)
    
    return loss

def compute_rewards(completions, answers, G):
    """简单的准确率奖励"""
    rewards = []
    for i, (comp, ans) in enumerate(zip(completions, 
                                         [a for a in answers for _ in range(G)])):
        predicted = extract_last_number(comp)
        if predicted is not None and abs(predicted - float(ans)) < 1e-5:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards)
```

### 小节 4：资源估算

```
模型: Qwen2.5-0.5B
训练方式: 全参数 + bf16
G = 8, batch_size = 2, max_seq_len = 512

显存估算:
├── 策略模型 (0.5B × 2 bytes bf16)     = ~1 GB
├── 参考模型 (0.5B × 2 bytes bf16)     = ~1 GB
├── 优化器状态 (0.5B × 8 bytes Adam)   = ~4 GB
├── 梯度 (0.5B × 2 bytes)             = ~1 GB
├── 激活值 (batch × seq × hidden)      = ~2 GB
├── 生成缓存 (KV cache for G samples)  = ~1 GB
└── 总计                               ≈ 10 GB ← RTX 3090 (24GB) ✅

模型: Qwen2.5-1.5B
同样设置:
└── 总计                               ≈ 20 GB ← RTX 3090 勉强 ✅ (需要 LoRA)
```

### 小节 5：在消费级 GPU 上的实践建议

| 策略 | 效果 | 代价 |
|------|------|------|
| 用更小模型（0.5B instead of 1.5B） | 显存减半 | 基线能力更弱 |
| LoRA 替代全参数 | 显存降 60% | 效果略差 |
| 减小 G（4 instead of 16） | 显存降 | 对比信号弱 |
| 减小 max_seq_length | 显存降 | 长推理被截断 |
| gradient_checkpointing | 显存降 30% | 训练变慢 |
| bf16 | 显存减半 | 无明显代价 |

## 测验题

### Q1（资源估算，4分）
你有一张 RTX 4090（24GB），要用 GRPO 训练 Qwen-1.5B，全参数 bf16，G=8。估算是否够用？如果不够，列出 2 个优化方案。

**答案**：1.5B 模型全参数 GRPO 约需 20-24GB，刚好在边缘，很可能 OOM。优化方案：(1) 使用 LoRA 降低到约 12GB；(2) 开启 gradient_checkpointing 减少激活值显存。（4分）

### Q2（对比分析，3分）
SimpleRL-Zoo 为什么不集成 vLLM？这对训练速度有什么影响？

**答案**：SimpleRL-Zoo 定位是小模型实验，0.5B-3B 模型的生成速度本身不慢，vLLM 的加速收益有限，引入 vLLM 反而增加了安装和调试复杂度。但对于 G=16+ 的场景，缺少 vLLM 会导致采样阶段成为瓶颈。（3分）

### Q3（实践理解，3分）
为什么 SimpleRL-Zoo 用 HF generate 而不是自己写采样循环？

**答案**：HF generate 已经实现了 top-p、temperature、repetition_penalty 等采样策略，自己写容易出 bug。SimpleRL-Zoo 的目标是简单可靠，复用成熟组件。（3分）
