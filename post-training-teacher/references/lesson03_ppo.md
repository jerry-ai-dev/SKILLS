# Lesson 3: PPO 算法深入

## 教学目标
理解 PPO 的设计动机，掌握 Clip 目标函数和 GAE，能读懂 PPO 的核心代码。

---

## 讲解要点

### 1. REINFORCE 的根本问题

REINFORCE 每次更新用**当前轨迹**的梯度，有两个问题：
1. **样本利用率低**：采样一条轨迹只用一次就扔掉
2. **步长不稳定**：梯度变化大时，一步更新可能破坏策略，之后就再也恢复不了

能不能把一批旧数据用**多次**？（Off-policy 学习）

---

### 2. 重要性采样（Importance Sampling）

用旧策略 $\pi_{\theta_\text{old}}$ 采集的数据，来估计当前策略 $\pi_\theta$ 的期望：

$$\mathbb{E}_{\tau \sim \pi_\theta}[f(\tau)] = \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}}\left[\frac{\pi_\theta(\tau)}{\pi_{\theta_\text{old}}(\tau)} f(\tau)\right]$$

**比率**（probability ratio）：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$$

- $r_t = 1$：新旧策略相同
- $r_t > 1$：新策略更倾向这个动作
- $r_t < 1$：新策略更不倾向这个动作

这样 REINFORCE 的 loss 变为：

$$\mathcal{L}_{\text{CPI}} = -\mathbb{E}_t\left[r_t(\theta) \cdot \hat{A}_t\right]$$

问题：如果 $r_t$ 很大（新旧策略差异大），梯度会爆炸，策略崩溃！这就是 TRPO 要解决的问题。

---

### 3. PPO-Clip 目标函数

PPO 用一个简单的 **clip 操作**来约束更新幅度：

$$\mathcal{L}^{\text{CLIP}}(\theta) = -\mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

- $\epsilon$ 通常取 0.1 或 0.2
- 当优势 $\hat{A}_t > 0$（好动作）：如果 $r_t$ 已经超过 $1+\epsilon$，就不再增加梯度（防止过度强化）
- 当优势 $\hat{A}_t < 0$（坏动作）：如果 $r_t$ 已经低于 $1-\epsilon$，就不再减少梯度（防止过度惩罚）

**直觉**：Clip 像一个安全边界，告诉模型"你跟老版本的差距最多允许在 ±ε 范围内，超出范围的梯度直接截断"。

---

### 4. 广义优势估计 GAE-λ

优势 $\hat{A}_t = Q(s_t, a_t) - V(s_t)$ 需要估计，有两种极端：
- **Monte Carlo（高方差低偏差）**：用完整轨迹回报减 $V(s_t)$
- **TD(1步，低方差高偏差）**：$\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

GAE-λ 用系数 $\lambda$ 来平衡两者：

$$\hat{A}_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

- $\lambda = 0$：退化为 1 步 TD（低方差高偏差）
- $\lambda = 1$：退化为 Monte Carlo（高方差低偏差）
- 实践中 $\lambda = 0.95$ 效果好

---

### 5. Actor-Critic 架构

PPO 需要两个组件：
- **Actor（策略网络）**：$\pi_\theta(a|s)$，负责选择动作
- **Critic（价值网络）**：$V_\phi(s)$，负责估计状态价值，用于计算优势

在 LLM RLHF 中：
- Actor = 要训练的语言模型
- Critic = 另一个语言模型（通常从同一个 checkpoint 初始化）
- **GRPO 的创新点**：去掉 Critic！用同一组生成数据中的相对奖励代替优势估计，极大节省显存

---

### 6. PPO 完整算法伪代码

```
初始化策略参数 θ，价值网络参数 φ
for epoch in range(N_epochs):
    # 采集数据
    用 π_θ 采集 T 步轨迹数据 {s_t, a_t, r_t}
    
    # 计算优势
    用 V_φ 计算 V(s_t)
    用 GAE-λ 计算 Â_t
    
    # 多次更新（关键！用同一批数据更新 K 次）
    for k in range(K_updates):
        计算 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        L_CLIP = -E[min(r_t * Â_t, clip(r_t, 1-ε, 1+ε) * Â_t)]
        L_VF   = E[(V_φ(s_t) - G_t)²]   # Critic loss
        L_ENT  = -E[entropy(π_θ)]         # 熵奖励（鼓励探索）
        
        loss = L_CLIP + c₁ * L_VF - c₂ * L_ENT
        梯度更新 θ 和 φ
    
    用新的 θ 替换 θ_old
```

---

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== PPO Clip Loss 核心实现 =====

def ppo_clip_loss(
    log_probs_new: torch.Tensor,     # 新策略的 log 概率 [batch]
    log_probs_old: torch.Tensor,     # 旧策略的 log 概率 [batch]（stop gradient）
    advantages:    torch.Tensor,     # 优势估计 [batch]
    epsilon: float = 0.2,            # clip 系数
) -> torch.Tensor:
    """
    PPO Clip 目标函数（最小化版本）
    
    公式:
    L = -E[ min(r * A, clip(r, 1-ε, 1+ε) * A) ]
    其中 r = π_new / π_old = exp(log_π_new - log_π_old)
    """
    # 计算概率比率（在 log 空间做减法，等价于 π_new / π_old）
    ratio = torch.exp(log_probs_new - log_probs_old.detach())
    # ratio shape: [batch]

    # 非 clip 目标
    obj_unclipped = ratio * advantages

    # clip 目标
    obj_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    # 取两者的较保守值（min）——这是 PPO 的关键！
    # 当优势 > 0（好动作）：min 限制 r 不能升太高（防止过度强化）
    # 当优势 < 0（坏动作）：min 限制 r 不能降太低（防止过度惩罚）
    loss = -torch.mean(torch.min(obj_unclipped, obj_clipped))

    return loss


# ===== 验证：理解 clip 的实际效果 =====
print("==== PPO Clip 效果可视化 ====\n")

# 模拟一批数据：8 条轨迹
batch_size = 8
torch.manual_seed(42)

log_probs_old = torch.randn(batch_size) - 1.0  # 旧策略 log 概率
log_probs_new = log_probs_old + torch.randn(batch_size) * 0.5  # 新策略（有偏差）
advantages    = torch.randn(batch_size)          # 优势值（有正有负）
ratio = torch.exp(log_probs_new - log_probs_old)

print(f"概率比率 r = π_new/π_old: {ratio.numpy().round(3)}")
print(f"优势 Â:                   {advantages.numpy().round(3)}")

# clip 的效果
clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)

# 哪些 ratio 被 clip 了？
clipped_mask = (ratio != clipped_ratio)
print(f"\n被 clip 掉的 ratio 数量: {clipped_mask.sum().item()}/{batch_size}")
print(f"（这些样本的梯度被截断，防止策略更新过大）")

loss = ppo_clip_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2)
print(f"\nPPO Clip Loss: {loss.item():.4f}")

# ===== GAE 实现 =====
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    广义优势估计 GAE-λ
    rewards: [T]   每步奖励
    values:  [T+1] 每步的 V(s)（最后一个是 V(s_T+1)，用 0 填充）
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        # TD 误差 δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t+1] - values[t]
        # GAE 递推: Â_t = δ_t + γλ * Â_{t+1}
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    return advantages

# 示例
T = 5
rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])  # 只有最后一步有奖励（如 LLM）
values  = torch.zeros(T + 1)  # 简单情况：价值估计为 0

gae_advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95)
print(f"\n==== GAE 演示（稀疏奖励，最后步得 1 分）====")
print(f"奖励:  {rewards.numpy()}")
print(f"GAE优势: {gae_advantages.numpy().round(4)}")
print("（优势从最后一步反向传播，越早的步骤折扣越多）")
```

---

## 测验题

**Q1（选择）** PPO-Clip 中 $\epsilon = 0.2$ 的物理含义是：
- A. 学习率为 0.2
- B. 新旧策略的概率比率被限制在 $[0.8, 1.2]$ 范围内
- C. 每次训练只用 20% 的数据
- D. 优势函数的最大值为 0.2

**答案**：B。$\text{clip}(r, 1-0.2, 1+0.2) = \text{clip}(r, 0.8, 1.2)$，把概率比率限制在 80%~120% 之间。

---

**Q2（概念）** 为什么 PPO 中 Critic 网络（价值网络）在 GRPO 中可以被去掉？GRPO 用什么替代优势估计？

**答案要点**：GRPO 对同一个 Prompt 生成 G 条候选回答，用这 G 条的奖励平均值作为 baseline，每条回答的优势 = 自身奖励 - 组内平均奖励，不需要单独训练一个参数量同等的价值网络，节省约一半显存。

---

**Q3（代码填空）** 下面是 PPO ratio 的计算，空格处应填什么？

```python
ratio = torch.exp(log_probs_new - _______)
```

**答案**：`log_probs_old.detach()`。`.detach()` 是必须的，旧策略的 log 概率是固定参考值，不需要对它求梯度。

---

## 课后练习（选做）
1. **手画**：在坐标轴上画出 PPO clip 目标 $\min(r \cdot A, \text{clip}(r, 0.8, 1.2) \cdot A)$ 关于 $r$ 的函数图像（分 $A>0$ 和 $A<0$ 两种情况）
2. **代码**：给 `compute_gae` 添加一条测试：当 $\lambda=0$ 时，GAE 应该退化为 1-step TD 优势；当 $\lambda=1$ 时，应该退化为 Monte Carlo 回报
