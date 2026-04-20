# Lesson 2: Policy Gradient & REINFORCE

## 教学目标
推导策略梯度定理，理解 REINFORCE 算法的原理、实现方式与局限性，为学习 PPO 打好基础。

---

## 讲解要点

### 1. 优化目标

我们想最大化：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[G(\tau)\right] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} r_t\right]$$

其中 $\tau = (s_0, a_0, r_0, s_1, a_1, \ldots)$ 是一条轨迹（trajectory）。

**直觉**：$J(\theta)$ 就是"用策略 $\pi_\theta$ 玩一把游戏的期望总奖励"，我们要对 $\theta$ 求梯度并上升。

---

### 2. 策略梯度定理（Policy Gradient Theorem）

核心问题：$G(\tau)$ 依赖于 $\pi_\theta$，但奖励函数本身不可微，怎么求梯度？

**对数导数技巧**（log-derivative trick）：

$$\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

因此：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log\pi_\theta(a_t|s_t) \cdot G_t\right]$$

**含义**：如果某条轨迹的回报 $G_t$ 很高，就增大这些动作的选择概率；如果回报低，就减小。

---

### 3. REINFORCE 算法

**算法步骤**：
1. 用当前策略 $\pi_\theta$ 采样一条轨迹 $\tau$
2. 计算每步的回报 $G_t$
3. 更新：$\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log\pi_\theta(a_t|s_t) \cdot G_t$

**对应的 loss（用梯度上升 → 等价为梯度下降取负号）**：

$$\mathcal{L}_{\text{REINFORCE}} = -\sum_t \log\pi_\theta(a_t|s_t) \cdot G_t$$

直觉：$\log\pi$ 是 LLM token 的对数概率，$G_t$ 是奖励权重。**高奖励的 token 要被强化（增大概率），低奖励的要被压制。**

---

### 4. REINFORCE 的问题：高方差

$G_t$ 是 Monte Carlo 估计，方差很大：
- 同样的动作，运气好时 $G_t$ 很高，运气差时很低
- 梯度更新方向每次都不同，学习很慢甚至发散

**解决方案：Baseline 技巧**

用一个基准 $b$（通常是状态价值估计）减去期望：

$$\mathcal{L} = -\sum_t \log\pi_\theta(a_t|s_t) \cdot (G_t - b)$$

- $G_t - b$ 被称为**优势 (Advantage)**：这个动作比平均水平好多少？
- $b$ 不影响梯度期望（无偏估计），但显著降低方差
- **GRPO 的精髓**正是在这里：用组内平均奖励作为 baseline

---

### 5. REINFORCE 与 LLM 的联系

在 LLM 中：
- $\log\pi_\theta(a_t|s_t)$ = 模型对第 $t$ 个 token 的对数概率（`log_softmax` 输出）
- $G_t$ = 整条生成文本的奖励（通常整条文本只有一个奖励）
- 训练 = 对每个 token 的对数概率做加权梯度更新

这正是 GRPO loss 的雏形！

---

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===== 玩具环境：CartPole（倒立摆） =====
# 需要 pip install gymnasium
try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("（未安装 gymnasium，跳过实际运行，仅展示 REINFORCE 核心逻辑）")

# ===== REINFORCE 核心逻辑 =====
class PolicyNet(nn.Module):
    """简单策略网络：输入状态 → 输出动作概率"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)  # 动作概率分布

def compute_returns(rewards, gamma=0.99):
    """计算每步的折扣回报 G_t（倒序累加）"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Baseline：减去均值，降低方差
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def reinforce_loss(log_probs, returns):
    """
    REINFORCE loss = -sum(log_prob * G_t)
    log_probs: 每步所选动作的对数概率 [T]
    returns:   每步的（归一化）回报    [T]
    """
    # 高奖励步骤的 log_prob 要增大 → 负号变成梯度上升
    return -(log_probs * returns).sum()

# ===== 伪运行演示（展示形状和逻辑） =====
print("==== REINFORCE 核心逻辑演示 ====\n")

# 假设一条轨迹有 10 步
T = 10
state_dim, action_dim = 4, 2  # CartPole 的状态和动作维度

policy = PolicyNet(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

# 模拟一条轨迹
fake_states  = torch.randn(T, state_dim)
fake_actions = torch.randint(0, action_dim, (T,))
fake_rewards = torch.randn(T)  # 模拟随机奖励

# 前向传播：得到每步动作的对数概率
log_probs = []
for t in range(T):
    probs = policy(fake_states[t])
    log_prob = torch.log(probs[fake_actions[t]])  # 选中动作的 log 概率
    log_probs.append(log_prob)
log_probs = torch.stack(log_probs)  # [T]

returns = compute_returns(fake_rewards.tolist(), gamma=0.99)

loss = reinforce_loss(log_probs, returns)
print(f"每步 log_prob    shape: {log_probs.shape}  (示例值: {log_probs[:3].detach().numpy().round(3)})")
print(f"每步 returns     shape: {returns.shape}   (归一化后)")
print(f"REINFORCE loss:  {loss.item():.4f}")

# 反向传播
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("\n参数已更新 ✅")

# ===== 关键对应关系 =====
print("\n==== 与 LLM 的对应关系 ====")
print("CartPole state  ↔  LLM 中的当前 token 上下文")
print("CartPole action ↔  LLM 下一个 token")
print("log_prob        ↔  model.log_softmax(...)[token_id]")
print("returns G_t     ↔  生成文本的奖励分数（如答案正确性）")
print("reinforce_loss  ↔  GRPO/PPO loss 的理论基础")
```

---

## 测验题

**Q1（选择）** 对数导数技巧（log-derivative trick）的目的是：
- A. 加快训练速度
- B. 让不可微的回报函数可以通过策略参数求导
- C. 减少采样数量
- D. 将强化学习转化为监督学习

**答案**：B。$R(\tau)$ 本身不依赖 $\theta$，log-derivative trick 把对 $P(\tau|\theta)$ 的梯度转化为 $\log\pi_\theta$ 的期望，从而可以用自动微分计算。

---

**Q2（推导）** 用一句话解释：为什么 REINFORCE 的 baseline $b$ 不影响梯度期望？

**答案要点**：因为 $\mathbb{E}[\nabla_\theta \log\pi_\theta(a|s) \cdot b] = b \cdot \nabla_\theta \underbrace{\sum_a \pi_\theta(a|s)}_{=1} = 0$，即 baseline 的梯度期望为零，只降低方差不引入偏差。

---

**Q3（代码）** 下面代码中的 `returns = (returns - returns.mean()) / (returns.std() + 1e-8)` 这行对应了哪个概念？去掉它会发生什么？

**答案**：对应 Baseline 技巧（减去均值相当于用均值做 baseline）。去掉后方差增大，训练会更不稳定，可能需要更多的探索步数才能收敛，在复杂任务上甚至无法收敛。

---

## 课后练习（选做）
1. **手推**：从 $\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_\tau[G(\tau)]$ 开始，一步步推导出 Policy Gradient Theorem
2. **代码**：安装 `gymnasium`，让 PolicyNet 在 CartPole 环境上真正跑起来，画出学习曲线
3. **思考**：为什么在 LLM 中，$G_t$ 通常用整条文本的奖励，而不是每个 token 单独的奖励？（提示：信用分配问题）
