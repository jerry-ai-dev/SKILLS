# Lesson 5: RLHF 完整流程

## 教学目标
理解 RLHF 三阶段流程（SFT → RM → RL），掌握 KL 惩罚项的作用，能描述 InstructGPT 的训练思路。

---

## 讲解要点

### 1. 为什么需要 RLHF？

问题：预训练的 LLM 只会"预测下一个 token"，不一定遵循人类的指令和价值观。

- 可能生成有害内容（因为训练数据包含有害内容）
- 可能答非所问（因为训练目标是补全，不是回答）
- 可能自信地给出错误答案（因为没有"不确定"的训练信号）

**RLHF（Reinforcement Learning from Human Feedback）** 是让 LLM "对齐" 人类偏好的方法。

---

### 2. RLHF 三阶段流程

**阶段 1：监督微调 (SFT)**
- 数据：人工标注的 (Prompt, 高质量 Response) 对
- 目标：让模型学会"遵循指令"的基本行为
- 输出：SFT 模型 $\pi_\text{SFT}$（同时保留一份冻结的参考模型 $\pi_\text{ref}$）

**阶段 2：训练奖励模型 (RM)**
- 数据：同一 Prompt 下多个回答的人类偏好排序
- 目标：学到 $r_\phi(x, y)$ 来模拟人类偏好打分
- 输出：奖励模型 $r_\phi$

**阶段 3：RL 优化 (PPO)**
- 数据：RM 给的分数 + KL 惩罚
- 目标：在 RM 的指引下最大化期望奖励
- 优化目标：

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(y|x) \; || \; \pi_\text{ref}(y|x))\right]$$

---

### 3. KL 散度惩罚项详解

**为什么需要 KL 惩罚？**

没有 KL 惩罚，RL 会让模型专门"骗" RM：
- 找到 RM 的盲点，生成分数高但质量差的文本
- 策略偏离 SFT 分布，语言能力退化

**KL 散度**衡量两个分布的差异：

$$\text{KL}(\pi_\theta || \pi_\text{ref}) = \sum_{y} \pi_\theta(y|x) \log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$$

- $\text{KL} = 0$：新策略和 SFT 模型完全相同
- $\text{KL}$ 越大：新策略偏离越多

**实现方式**：在奖励中减去 KL 惩罚：

$$r_{\text{effective}}(x, y) = r_\phi(x, y) - \beta \cdot \log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$$

- $\beta$ 越大：越保守（接近 SFT 模型）
- $\beta$ 越小：自由度越大（风险越高）

---

### 4. InstructGPT / ChatGPT 的对应

InstructGPT (2022) 是第一个大规模 RLHF LLM：
- SFT 阶段：约 12k condition demonstration 数据（人类示范）
- RM 阶段：约 33k comparison 数据（两个回答谁更好）
- RL 阶段：约 31k 无标签 Prompt，用 PPO 优化

GPT-4 / ChatGPT 用了类似但扩大规模的流程。

---

### 5. 你的项目 vs RLHF

**你即将做的项目（数学推理）的简化**：

| RLHF 标准流程 | 你的项目 |
|------------|--------|
| 人类偏好标注 RM | 规则奖励（答案对 +1，错 0） |
| PPO + Critic | GRPO（无 Critic） |
| 对齐安全偏好 | 提升数学推理能力 |
| KL 惩罚 | 同样有 KL 惩罚 |

**关键点**：你的项目用"答案是否正确"作为奖励，避免了训练 RM 的复杂性，这正是 DeepSeek R1 的设计思路。

---

## 代码示例

```python
import torch
import torch.nn.functional as F

# ===== 模拟 RLHF 中的 Effective Reward 计算 =====
# 在实际实现中，这是 PPO 更新前的关键步骤

def compute_effective_reward(
    reward_scores:   torch.Tensor,   # RM 给的原始分数 [batch]
    log_probs_new:   torch.Tensor,   # 当前策略的 token log概率  [batch, seq_len]
    log_probs_ref:   torch.Tensor,   # 参考(SFT)模型的 log概率    [batch, seq_len]
    beta: float = 0.1,               # KL 惩罚系数
) -> torch.Tensor:
    """
    effective_reward = r_phi(x,y) - beta * KL(π_θ || π_ref)

    KL 散度在 token 级别的近似：
    KL(π_θ || π_ref) ≈ mean_t [ log π_θ(a_t) - log π_ref(a_t) ]
    """
    # token 级 KL 差值，对序列取均值
    token_kl  = log_probs_new - log_probs_ref       # [batch, seq_len]
    kl_penalty = token_kl.mean(dim=-1)               # [batch]

    effective_rewards = reward_scores - beta * kl_penalty
    return effective_rewards


# ===== 示例 =====
print("==== RLHF Effective Reward 计算示意 ====\n")

batch_size, seq_len = 4, 10
torch.manual_seed(0)

# 模拟 RM 分数（来自 Reward Model）
rm_scores = torch.tensor([1.2, -0.5, 0.8, 0.1])

# 模拟 token log 概率（来自当前策略和参考模型）
log_probs_new = torch.randn(batch_size, seq_len) - 1.0  # 当前策略
log_probs_ref = torch.randn(batch_size, seq_len) - 1.0  # 参考 SFT 模型

# KL 较大 = 当前策略偏离参考模型较多
print(f"RM 原始分数:       {rm_scores.numpy()}")

kl_approx = (log_probs_new - log_probs_ref).mean(dim=-1)
print(f"KL 近似值:         {kl_approx.detach().numpy().round(4)}")

for beta in [0.0, 0.1, 0.5]:
    eff = compute_effective_reward(rm_scores, log_probs_new, log_probs_ref, beta=beta)
    print(f"β={beta}  有效奖励:  {eff.detach().numpy().round(4)}")

print("\n注意：β 越大，KL 惩罚越强，有效奖励与 RM 原始分数差异越大")

# ===== RLHF 三阶段流程图（用代码语言描述） =====
print("\n==== RLHF 三阶段数据流（简化版） ====")
stages = [
    ("阶段1 SFT",  "Input: (prompt, expert_response)  → 训练目标: CE loss  → 输出: π_SFT"),
    ("阶段2 RM",   "Input: (prompt, y_chosen, y_rejected) → 训练目标: preference loss → 输出: r_φ"),
    ("阶段3 PPO",  "Input: prompt → π_θ采样回答y → r_φ打分 → KL惩罚 → PPO更新 → 输出: π_θ*"),
]
for name, desc in stages:
    print(f"\n  {name}:")
    print(f"    {desc}")
```

---

## 测验题

**Q1（选择）** RLHF 中 KL 散度惩罚 $\beta \cdot \text{KL}(\pi_\theta || \pi_\text{ref})$ 的主要目的是：
- A. 加快训练速度
- B. 防止模型偏离 SFT 分布太远，避免 Reward Hacking
- C. 减少 Reward Model 的训练数据需求
- D. 让模型生成更多样化的回答

**答案**：B。KL 惩罚限制了策略的更新范围，防止模型专门"骗"奖励模型。

---

**Q2（排序）** 请将 RLHF 的步骤按正确顺序排列：
- A. 用 PPO 根据 RM 分数更新策略
- B. 收集人类偏好对数据，训练 Reward Model
- C. 在 Prompt-Response 对上监督微调基础模型

**答案**：C → B → A（SFT → RM → PPO）

---

**Q3（开放）** 在你的数学推理项目中，不需要训练 Reward Model，而是直接用"答案是否正确"作为奖励。这样做有什么优点？有什么缺点？

**答案要点**：
- 优点：不需要人工标注偏好数据；奖励不会 hack（答案对就是对）；实现简单
- 缺点：无法给推理**过程**打分（只管结果对不对）；对于需要格式或安全性对齐的任务不够用；可能奖励"猜答案"的行为

---

## 课后练习（选做）
1. **图示**：用 `renderMermaidDiagram`（在上课时）或手画一幅 RLHF 三阶段的数据流程图
2. **阅读**：InstructGPT 论文摘要（[https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)），重点看 Figure 2（三阶段图示）
