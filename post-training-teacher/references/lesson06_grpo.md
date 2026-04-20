# Lesson 6: GRPO 算法

## 教学目标
掌握 GRPO 的设计动机、完整 loss 公式、与 PPO 的核心区别，能用 PyTorch 实现 GRPO loss（约 20 行）。

---

## 讲解要点

### 1. PPO 在 LLM 训练中的瓶颈

PPO 需要：
- **Actor**（当前策略，正在训练）
- **Critic**（价值网络，与 Actor 同等大小）
- **Reference model**（SFT 模型，冻结）
- **Reward model**（打分，可能也是 LLM 级别）

训练一个 7B LLM 用 PPO = 同时在内存里放 ~4 个 7B 模型。显存极其紧张！

**GRPO**（Group Relative Policy Optimization）的目标：**去掉 Critic**，用更简单的方法估计优势。

---

### 2. GRPO 的核心思路：组内相对奖励

对同一个 Prompt $x$，用当前策略采样 $G$ 条回答：$\{y_1, y_2, \ldots, y_G\}$

每条回答从 Reward Model 或规则函数得到奖励：$\{r_1, r_2, \ldots, r_G\}$

**组内归一化**（group-relative advantage）：

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1,\ldots,r_G)}{\text{std}(r_1,\ldots,r_G) + \epsilon}$$

**直觉**：不需要绝对分数的高低，只需要"在这一组里，这条回答比平均水平好多少"。这正是 REINFORCE 中 baseline 技巧的一个变体——用组内均值做 baseline！

---

### 3. GRPO 完整 Loss 公式

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left[\min\left(r_{t,i}(\theta) \hat{A}_i,\; \text{clip}(r_{t,i}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right) - \beta \cdot \text{KL}_t\right]$$

其中：
- $r_{t,i}(\theta) = \frac{\pi_\theta(a_{t,i}|s_{t,i})}{\pi_{\theta_\text{old}}(a_{t,i}|s_{t,i})}$：第 $i$ 条回答第 $t$ 个 token 的概率比率
- $\hat{A}_i$：第 $i$ 条回答的组内归一化优势（整条回答共用同一个优势值）
- $\beta \cdot \text{KL}_t$：per-token KL 惩罚，防止偏离参考模型
- 分母 $|y_i|$：对序列长度做归一化，防止长回答主导梯度

---

### 4. GRPO vs PPO 对比

| 特性 | PPO | GRPO |
|------|-----|------|
| 是否需要 Critic | ✅ 需要 (同等大小) | ❌ 不需要 |
| 优势估计 | GAE (需要 V(s)) | 组内奖励归一化 |
| 显存占用 | ~4x 模型大小 | ~2x 模型大小 |
| 偏差-方差 | 低偏差高方差控制 | 相对较高方差但实践有效 |
| 适合场景 | 游戏等稠密奖励 | LLM 稀疏奖励（答案对不对） |
| DeepSeek R1 | — | ✅ 使用 GRPO |

---

### 5. 为什么 GRPO 适合数学推理？

数学题的奖励天然是"分组比较"友好的：
- 同一道题生成 8 条解题过程
- 部分对（$r=1$）、部分错（$r=0$）
- 组内归一化后，正确答案的优势 > 0，错误答案的优势 < 0
- PPO clip 保证每步更新不过激

这就是 DeepSeek R1 用 GRPO + 数学验证器的设计逻辑！

---

## 代码示例

```python
import torch
import torch.nn.functional as F

# ===== GRPO Loss 核心实现 =====

def grpo_advantage(rewards: torch.Tensor) -> torch.Tensor:
    """
    组内相对优势归一化
    rewards: [G] 同一个 prompt 下 G 条回答的奖励
    返回:    [G] 每条回答的归一化优势
    """
    mean = rewards.mean()
    std  = rewards.std() + 1e-8
    return (rewards - mean) / std


def grpo_loss(
    log_probs_new:  torch.Tensor,  # 新策略的 token log 概率 [G, T]
    log_probs_old:  torch.Tensor,  # 旧策略的 token log 概率 [G, T]（stop grad）
    log_probs_ref:  torch.Tensor,  # 参考模型的 token log 概率 [G, T]（stop grad）
    rewards:        torch.Tensor,  # 每条回答的奖励 [G]
    epsilon: float = 0.2,
    beta:    float = 0.1,
) -> torch.Tensor:
    """
    GRPO Loss

    G: 每个 prompt 生成的候选回答数
    T: 序列长度（简化假设所有回答等长，实际中用 mask 处理）
    """
    G, T = log_probs_new.shape

    # 1. 组内相对优势
    advantages = grpo_advantage(rewards)           # [G]
    advantages = advantages.unsqueeze(-1).expand(G, T)  # [G, T]，每步共用同一优势

    # 2. 概率比率 r_t = π_new / π_old（log 空间相减）
    ratio = torch.exp(log_probs_new - log_probs_old.detach())  # [G, T]

    # 3. PPO Clip 目标
    obj_unclipped = ratio * advantages
    obj_clipped   = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_obj    = torch.min(obj_unclipped, obj_clipped)      # [G, T]

    # 4. per-token KL 惩罚（当前策略 vs 参考模型）
    kl_penalty = log_probs_new - log_probs_ref.detach()        # [G, T]

    # 5. 合并：对每条回答的所有 token 取均值，再对 G 条回答取均值
    token_loss = -(policy_obj - beta * kl_penalty)             # [G, T]（取负号变最小化）
    loss = token_loss.mean()

    return loss


# ===== 示例运行 =====
print("==== GRPO Loss 演示 ====\n")

G, T = 4, 8   # 4 条回答（同一 prompt），每条 8 个 token
torch.manual_seed(42)

log_probs_old = torch.randn(G, T) - 2.0
log_probs_new = log_probs_old + torch.randn(G, T) * 0.3
log_probs_ref = log_probs_old.clone()

# 模拟数学题奖励：2 条正确（1.0），2 条错误（0.0）
rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])

advantages = grpo_advantage(rewards)
print(f"奖励:    {rewards.numpy()}")
print(f"组内优势: {advantages.numpy().round(4)}")
print(f"  → 正确答案优势 > 0，错误答案优势 < 0\n")

loss = grpo_loss(log_probs_new, log_probs_old, log_probs_ref, rewards, epsilon=0.2, beta=0.1)
print(f"GRPO Loss: {loss.item():.4f}")

# ===== 和 PPO 的直观对比 =====
print("\n==== GRPO vs PPO：需要加载的模型 ====")
print("PPO: Actor(7B) + Critic(7B) + RefModel(7B) + RewardModel(7B) ≈ 28B 参数显存")
print("GRPO: Actor(7B) + OldActor(7B) + RefModel(7B) ≈ 21B 参数显存（省 25%）")
print("     实际还可以让 OldActor 和 Actor 共享（延迟更新），省更多显存")
```

---

## 测验题

**Q1（选择）** GRPO 相比 PPO 最大的工程优势是：
- A. 训练速度快一倍
- B. 不需要训练 Critic（价值网络），节省显存
- C. 不需要参考模型（SFT 模型）
- D. 奖励模型更准确

**答案**：B。去掉 Critic 是 GRPO 最核心的工程改进，在显存受限的大模型训练中意义重大。

---

**Q2（填空）** GRPO 组内优势的计算公式为：
$$\hat{A}_i = \frac{r_i - \_\_\_}{\_\_\_ + \epsilon}$$

**答案**：$\hat{A}_i = \frac{r_i - \text{mean}(r_1,\ldots,r_G)}{\text{std}(r_1,\ldots,r_G) + \epsilon}$

---

**Q3（代码）** 在 `grpo_loss` 函数中，为什么 `log_probs_old.detach()` 是必要的？

**答案**：`log_probs_old` 是旧策略的 token 概率，是**固定的参考值**（采样数据时用旧策略得到的），不应该参与当前策略 $\theta$ 的梯度计算。如果不 detach，反向传播会错误地通过 `log_probs_old` 更新参数，导致 ratio 计算混乱。

---

## 课后练习（选做）
1. **推导**：从 REINFORCE loss 出发，加入 baseline 和 PPO clip，一步步推导出 GRPO loss
2. **代码**：修改 `grpo_loss`，支持变长序列（用 `attention_mask` 做 token 级 mask，对有效 token 取均值）
3. **思考**：如果 G 组内所有 reward 都相同（比如全部答对或全部答错），优势会是多少？这对训练有什么影响？
