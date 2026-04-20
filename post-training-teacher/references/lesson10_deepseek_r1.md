# Lesson 10: DeepSeek R1 论文精读

## 教学目标
读懂 DeepSeek R1 的训练流程，理解为什么纯 RL 能涌现推理能力，能对比 R1 与 InstructGPT 的异同，为独立复现项目做好准备。

---

## 讲解要点

### 1. 论文背景：为什么 R1 重要？

2024 年之前，增强 LLM 推理能力的主流方法：
- 精心设计的 Chain-of-Thought (CoT) 提示词
- 监督学习 CoT 示范数据

**DeepSeek R1 的惊人发现（2025 年 1 月）**：
- **R1-Zero**：在没有 SFT CoT 数据的情况下，**仅用 RL** 就能让模型学会复杂推理！
- 模型自主涌现出"反思"行为：`"Wait, let me reconsider..."` 完全没有人工监督这种行为
- 模型生成的思维链越来越长（但内容越来越有效）

---

### 2. R1-Zero：纯 RL 的涌现推理

**训练设置**：
- 基础模型：DeepSeek-V3-Base（预训练，无 SFT）
- 强化学习算法：**GRPO**（你在 Lesson 6 学过的！）
- 奖励函数：**规则奖励**，无需 Reward Model

**两种奖励**：
1. **准确率奖励**：数学题用验证器检查答案，代码题用编译/测试通过率
2. **格式奖励**：是否使用 `<think>...</think>` 和 `<answer>...</answer>` 标签

$$r(y) = r_{\text{accuracy}}(y) + r_{\text{format}}(y)$$

**R1-Zero 的问题**：
- 可读性差（英中混杂，格式混乱）
- 没有进行"冷启动"的 SFT，初期训练不稳定

---

### 3. DeepSeek R1：完整四阶段流程

```
阶段 1: SFT Cold Start
  ↓ 用少量高质量 CoT 数据（长思维链格式）做 SFT
  ↓ 让模型先学会"怎么格式化推理过程"
  
阶段 2: GRPO 推理增强（RL 第一轮）
  ↓ 用 GRPO + 规则奖励训练，专注提升数学/代码推理
  ↓ 结果：推理能力大幅提升，但通用能力可能有退化

阶段 3: Rejection Sampling + SFT（拒绝采样微调）
  ↓ 用阶段 2 的模型生成大量回答，只保留正确且格式好的
  ↓ 结合通用对话数据，做 SFT 恢复通用能力

阶段 4: GRPO 全面对齐（RL 第二轮）
  ↓ 同时优化推理 + 通用对话 + 安全性
  ↓ 输出：DeepSeek R1
```

---

### 4. 关键技术细节

**GRPO 配置**：
- 每个 Prompt 生成 $G = 16$ 条候选回答
- $\epsilon = 0.2$（clip 范围）
- $\beta = 0.1$（KL 惩罚系数）
- 最大序列长度：32k token（思维链可以很长！）

**"Aha Moment"（自我反思涌现）**：

论文发现，训练到一定程度时，模型开始在思维链中出现自我纠正：
```
<think>
Let me compute 15 × 24...
First approach: 15 × 24 = 15 × 20 + 15 × 4 = 300 + 60 = 360
Wait, let me verify: 24 × 15 = 24 × 10 + 24 × 5 = 240 + 120 = 360 ✓
</think>
<answer>360</answer>
```

这种行为**完全是涌现的**，没有任何训练数据包含"Wait, let me reconsider"这样的模式。

---

### 5. R1 vs InstructGPT 对比

| 维度 | InstructGPT / ChatGPT | DeepSeek R1 |
|------|----------------------|-------------|
| 奖励来源 | 人类偏好标注（Reward Model） | 规则验证器（答案对不对） |
| RL 算法 | PPO | GRPO |
| 需要 Reward Model | ✅ 是（大工程） | ❌ 不需要（简化很多） |
| 需要 Critic | ✅ 是（PPO + Critic） | ❌ 不需要（GRPO） |
| 目标 | 对齐价值观/安全/指令跟随 | 增强推理能力 |
| 训练阶段 | 3 阶段（SFT→RM→PPO） | 4 阶段（SFT→GRPO→RS-SFT→GRPO） |

---

### 6. 你的项目对应 R1 的哪个部分？

你要做的 **简化版 R1**：
```
SFT（用数学 QA 数据）→ GRPO（用答案正确性作奖励）
```

对应 R1 的：阶段 1 + 阶段 2 的核心逻辑

区别：
- 不需要实现 Rejection Sampling（阶段 3）
- 不需要安全性对齐（阶段 4）
- 模型更小（Qwen-0.5B 或 1.5B，而非 DeepSeek 671B）

---

## 代码示例

```python
# ===== 整合：模拟一个完整的 R1 训练步骤（简化版）=====
import torch
import torch.nn.functional as F

print("==== R1 训练步骤模拟（概念级演示）====\n")

# 场景：训练模型解数学题 "What is 15 × 24?"
# 模型生成 G=4 条候选回答（正常应为 G=16）

G = 4  # 候选回答数量

# 模拟 tokenizer 和模型输出（用随机数代替）
# 真实实现中这些是 model.generate() 的结果
torch.manual_seed(0)
seq_len = 20  # 假设每条回答 20 个 token

log_probs_new = torch.randn(G, seq_len) - 2.0
log_probs_old = log_probs_new.clone().detach()
log_probs_ref = log_probs_new.clone().detach() + torch.randn(G, seq_len) * 0.1

# ===== 奖励函数：规则验证 =====
def check_answer(generated_text: str, correct_answer: str) -> float:
    """
    简化的奖励函数：检查答案是否正确
    真实版本会用 sympy 或代码执行器验证
    """
    # 尝试从 <answer>...</answer> 中提取答案
    import re
    match = re.search(r'<answer>(.*?)</answer>', generated_text)
    if match:
        predicted = match.group(1).strip()
        return 1.0 if predicted == correct_answer else 0.0
    return 0.0  # 格式错误，0 分


# 模拟 4 条生成的回答（2 正确，2 错误）
generated_texts = [
    "<think>15×24 = 360</think><answer>360</answer>",       # 正确
    "<think>15×24 = 350</think><answer>350</answer>",       # 错误
    "<think>Wait... 15×20=300, 15×4=60, total=360</think><answer>360</answer>",  # 正确（有反思！）
    "<think>15+24=39</think><answer>39</answer>",           # 错误
]
correct = "360"

rewards = torch.tensor([check_answer(t, correct) for t in generated_texts])
print(f"生成的回答:")
for i, (text, r) in enumerate(zip(generated_texts, rewards.tolist())):
    print(f"  [{i}] 奖励={r:.0f}  {text[:50]}...")

# ===== GRPO 优势计算 =====
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
print(f"\n组内奖励: {rewards.numpy()}")
print(f"归一化优势: {advantages.numpy().round(4)}")
print(f"正确答案优势 > 0 ✅，错误答案优势 < 0 ✅")

# ===== 计算 GRPO Loss =====
adv_expanded = advantages.unsqueeze(-1).expand(G, seq_len)
ratio = torch.exp(log_probs_new - log_probs_old.detach())
obj = torch.min(ratio * adv_expanded,
                torch.clamp(ratio, 0.8, 1.2) * adv_expanded)
kl_pen = log_probs_new - log_probs_ref.detach()
loss = -(obj - 0.1 * kl_pen).mean()
print(f"\nGRPO Loss: {loss.item():.4f}")

# ===== 宏观训练流程总结 =====
print("\n==== 你的 R1 简化版 完整训练计划 ====")
steps = [
    "Step 1: 准备数学推理数据集（如 GSM8K）",
    "Step 2: SFT Cold Start（用 CoT 格式数据微调 Qwen-0.5B）",
    "Step 3: GRPO 训练循环：",
    "   3a: 对每道题生成 G 条解答",
    "   3b: 用 sympy 验证答案计算奖励",
    "   3c: 归一化优势，计算 GRPO loss",
    "   3d: 反向传播，更新模型",
    "Step 4: 在测试集上评估 Pass@1 准确率",
]
for step in steps:
    print(f"  {step}")
```

---

## 测验题

**Q1（选择）** DeepSeek R1-Zero 最令人惊讶的发现是：
- A. 用了很多高质量 CoT 训练数据
- B. 仅用强化学习（无 SFT）就涌现出自我反思和推理能力
- C. 训练速度比 GPT-4 快 10 倍
- D. 首次证明 bf16 训练优于 fp32

**答案**：B。R1-Zero 在没有任何 CoT 示范数据的情况下，仅凭"答案对不对"的奖励信号，让模型涌现出了复杂的推理行为，这打破了之前认为"推理需要大量人工标注 CoT 数据"的认知。

---

**Q2（对比）** 你的项目与 DeepSeek R1 的最大区别是什么？（写出 2 点）

**答案示例**：
1. 模型规模：你用 Qwen-0.5B 或 1.5B，R1 用 DeepSeek-V3 671B
2. 无需 Rejection Sampling（阶段 3）和安全对齐（阶段 4），只做核心的 SFT + GRPO 两阶段
3. 奖励函数相同（规则验证），RL 算法相同（GRPO）——这是核心共同点

---

**Q3（设计）** 如果你要验证"模型是否真的学会了推理而不是记忆答案"，你会设计什么样的测试？

**答案要点**：
- 用模型**从未见过的**数学题（OOD 测试集）评估
- 对比 SFT 基线 vs GRPO 训练后的正确率提升
- 检查思维链的质量：是否有有效的中间步骤，而非直接跳到答案
- 测试"错误诱导"：给出一道题的错误前几步，看模型能否发现并纠正

---

## 课后练习（选做）

1. **精读**：阅读 DeepSeek R1 论文的 Section 2.1（R1-Zero 训练流程）和 Section 3（主要结果），重点看 Figure 3（"Aha Moment" 的例子）
   - 论文链接：https://arxiv.org/abs/2501.12948
2. **对比**：用表格整理 InstructGPT、R1-Zero、DeepSeek R1 三个方法的训练阶段、奖励来源、是否需要 RM、使用的 RL 算法
