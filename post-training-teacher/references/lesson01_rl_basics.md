# Lesson 1: 强化学习基础 & MDP

## 教学目标
理解强化学习的核心概念，能用 MDP 框架描述语言模型的训练问题。

---

## 讲解要点

### 1. 为什么语言模型需要强化学习？

监督学习（SFT）能做什么：
- 给模型大量"正确答案"，让它模仿
- 问题：**谁来提供"正确答案"？** 对于数学推理，人类专家很难写出所有正确思路

强化学习能做什么：
- 不需要"答案"，只需要一个**评分器（奖励函数）**
- 模型自己探索，高分路径被强化，低分路径被压制
- 类比：学围棋不靠背棋谱，靠和对手下棋获得输赢反馈

**在 LLM 中的应用**：
- 奖励函数可以是：验证器（答案对不对）、人类偏好（更喜欢哪个回答）、风格规则
- 模型通过生成不同回答并获得奖励，学会更好地推理

---

### 2. 马尔可夫决策过程 (MDP)

MDP 由五元组 $(S, A, P, R, \gamma)$ 定义：

$$\text{MDP} = (S, A, P, R, \gamma)$$

- $S$：状态空间 (State space)
- $A$：动作空间 (Action space)
- $P(s'|s,a)$：状态转移概率
- $R(s, a)$：奖励函数，执行动作 $a$ 后获得的即时奖励
- $\gamma \in [0,1)$：折扣因子，控制未来奖励的重要程度

**LLM 对应关系**：

| MDP 概念 | LLM 中的含义 |
|---------|------------|
| 状态 $s_t$ | 当前已生成的 token 序列（上下文） |
| 动作 $a_t$ | 下一个生成的 token |
| 转移 $P(s'|s,a)$ | 确定性的（append token） |
| 奖励 $R$ | 在序列结束后给出（如答案正确 +1） |
| 轨迹 | 完整生成的一条文本序列 |

---

### 3. 策略函数 (Policy)

策略 $\pi_\theta(a|s)$ 是在状态 $s$ 下选择动作 $a$ 的概率分布：

$$\pi_\theta(a|s) = P(a_t = a \mid s_t = s; \theta)$$

- 对于 LLM：$\pi_\theta(\text{next token} \mid \text{context})$ 就是 Softmax 输出
- $\theta$：模型的可训练参数
- **训练目标**：找到最优 $\theta^*$，使期望累积奖励最大

---

### 4. 回报 (Return) 与价值函数 (Value Function)

**回报（轨迹总奖励）**：
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

- 折扣因子 $\gamma$ 让近期奖励更重要  
- $\gamma = 0$：只看当前奖励；$\gamma \to 1$：远期奖励同等重要

**状态价值函数**（预期回报）：
$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$$

**Q 函数**（动作-状态价值）：
$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$$

---

### 5. LLM 训练中的奖励特点

与游戏 RL 的区别：
- 奖励通常是**稀疏的**：只在生成完整回答后才给分（非每步都有奖励）
- 动作空间巨大：词表大小通常 30k-100k+
- 序列可能很长（数百到数千 token）

这就是为什么朴素 RL 不稳定，需要 PPO/GRPO 等改进算法。

---

## 代码示例

```python
import numpy as np
import random

# ===== Mini Bandit 环境 =====
# 场景：你面前有 K 个"老虎机"，每个机器每次拉动给一个随机奖励
# 目标：在 T 次拉动内，最大化总奖励（只能拉，不能提前知道真实期望）

class BanditEnv:
    """K 臂老虎机"""
    def __init__(self, k=5):
        # 每个机器的真实期望奖励（模型不知道）
        self.k = k
        self.true_rewards = np.random.normal(loc=0.0, scale=1.0, size=k)
        print(f"各机器真实期望奖励（模型不可见）: {self.true_rewards.round(2)}")

    def pull(self, arm: int) -> float:
        """拉动第 arm 个机器，得到含噪声的奖励"""
        return self.true_rewards[arm] + np.random.normal(0, 0.5)

# ===== 贪心策略（利用已知最优） =====
def greedy_policy(estimates, n_step):
    """始终选择当前估计奖励最高的机器"""
    return np.argmax(estimates)

# ===== ε-贪心策略（平衡探索与利用） =====
def epsilon_greedy_policy(estimates, epsilon=0.1):
    """以 epsilon 概率随机探索，1-epsilon 概率选最优"""
    if random.random() < epsilon:
        return random.randint(0, len(estimates) - 1)
    return np.argmax(estimates)

def run_bandit(env, policy_fn, steps=1000):
    k = env.k
    estimates = np.zeros(k)   # 各机器奖励的估计值
    counts    = np.zeros(k)   # 各机器被拉的次数
    total_reward = 0

    for t in range(steps):
        arm = policy_fn(estimates, t)
        reward = env.pull(arm)

        # 增量更新估计均值
        counts[arm] += 1
        estimates[arm] += (reward - estimates[arm]) / counts[arm]
        total_reward += reward

    return total_reward

# 对比两种策略
env = BanditEnv(k=5)
r_greedy = run_bandit(env, greedy_policy, steps=1000)
r_eps    = run_bandit(env, lambda e, t: epsilon_greedy_policy(e, epsilon=0.1), steps=1000)

print(f"\n贪心策略总奖励:       {r_greedy:.1f}")
print(f"ε-贪心策略总奖励 (ε=0.1): {r_eps:.1f}")
print(f"\n最优机器: {np.argmax(env.true_rewards)}（奖励 {max(env.true_rewards):.2f}）")
print("结论: ε-贪心通常总奖励更高，因为它能探索发现真正最优的机器")
```

---

## 测验题

**Q1（选择）** MDP 中的"折扣因子" $\gamma = 0.9$，表示：
- A. 只考虑当前步的奖励
- B. 100 步后的奖励 = $(0.9)^{100} \approx 0.0000265$ 倍的即时奖励
- C. 每步奖励都乘 0.9
- D. 策略更新步长为 0.9

**答案**：B。折扣因子让未来奖励呈指数衰减，远期收益重要性降低。

---

**Q2（简答）** 为什么语言模型生成一段文字可以被建模为 MDP？请对应说明状态、动作、奖励分别是什么。

**答案要点**：
- 状态：已生成的 token 序列（当前上下文）
- 动作：从词表中选择下一个 token（Softmax 输出分布中采样）
- 奖励：稀疏奖励，通常在生成结束后才给出（对整条文本打分）

---

**Q3（推导）** 若 $\gamma = 0.9$，三步奖励分别为 $R_1=1, R_2=2, R_3=3$，求 $t=0$ 时刻的回报 $G_0$。

**答案**：
$$G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 = 1 + 0.9 \times 2 + 0.81 \times 3 = 1 + 1.8 + 2.43 = 5.23$$

---

## 课后练习（选做）
1. 修改 Bandit 代码，画出两种策略随步数增加的累积奖励曲线（用 matplotlib）
2. 思考：如果语言模型生成每个 token 都能得到一个奖励，会有什么问题？（提示：信用分配）
