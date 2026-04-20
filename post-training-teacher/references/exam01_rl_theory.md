# Exam 1: RL 理论基础阶段考试

## 考试说明
- **范围**：Lesson 1-3（MDP、Policy Gradient、PPO）
- **题数**：10 题，满分 100 分
- **分布**：选择题 4 道（每题 8 分）+ 推导/简答题 3 道（每题 10 分）+ 代码理解题 3 道（每题 6 分）
- **规则**：逐题作答，尽量不翻笔记

---

## 题目

### Q1【选择，8分】
在 LLM 的 RL 训练框架下，下列哪项对应 MDP 中的"动作 (Action)"？

A. 当前已生成的 token 序列  
B. 从词表中选择并生成的下一个 token  
C. Reward Model 给出的分数  
D. Transformer 的隐层表示  

**答案**：B  
**评分标准**：选 B 得 8 分；其余 0 分  
**解析**：状态=已生成的token序列，动作=下一个token的选择，奖励=来自RM或规则函数的分数

---

### Q2【选择，8分】
REINFORCE 算法更新公式 $\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log\pi_\theta(a_t|s_t) \cdot G_t$ 中，当某条轨迹的 $G_t$ 为负数时，更新效果是：

A. 增大选择动作 $a_t$ 的概率  
B. 不改变策略  
C. 减小选择动作 $a_t$ 的概率  
D. 重置参数到初始值  

**答案**：C  
**评分标准**：选 C 得 8 分  
**解析**：$G_t < 0$ 时，梯度方向反向，参数更新使 $\log\pi_\theta(a_t|s_t)$ 减小，即降低该动作的概率

---

### Q3【选择，8分】
PPO-Clip 中，$\epsilon = 0.2$ 的物理含义是：

A. 学习率上限为 0.2  
B. 新旧策略的概率比率 $r_t = \pi_\theta / \pi_{\theta_\text{old}}$ 被限制在 $[0.8, 1.2]$  
C. 每次只更新 20% 的参数  
D. 优势函数值不超过 0.2  

**答案**：B  
**评分标准**：选 B 得 8 分

---

### Q4【选择，8分】
GAE-λ 中，当 $\lambda = 0$ 时，优势估计退化为：

A. Monte Carlo 回报  
B. 1步 TD 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$  
C. $r_t$ 本身（原始奖励）  
D. 零向量  

**答案**：B  
**评分标准**：选 B 得 8 分  
**解析**：$\lambda=0$ 时 GAE 只用当前一步的 TD 误差，不向后累积，即退化为1步TD优势

---

### Q5【推导，10分】
解释"对数导数技巧"（log-derivative trick）：为什么 $\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ 可以写成 $\mathbb{E}_{\tau \sim \pi_\theta}[R(\tau) \cdot \nabla_\theta \log P(\tau|\theta)]$？写出关键步骤。

**答案（评分标准）**：
- 写出 $\nabla_\theta \mathbb{E}[R] = \nabla_\theta \int R(\tau) P(\tau|\theta) d\tau$（2分）
- 用 $\nabla_\theta P = P \cdot \nabla_\theta \log P$ 变形（4分）
- 最终写出期望形式 $= \mathbb{E}_\tau[R(\tau) \nabla_\theta \log P(\tau|\theta)]$（2分）
- 说明 $\log P(\tau|\theta) = \sum_t \log \pi_\theta(a_t|s_t) +$ 与θ无关的项（2分）

---

### Q6【简答，10分】
什么是 Reward Hacking？在 RLHF 中如何用 KL 惩罚项防止它？（4-6句话）

**答案（评分标准）**：
- 正确定义 Reward Hacking：模型找到让 RM 打高分的捷径而非真正提升质量（4分）
- KL 散度衡量新策略与参考模型分布差异（3分）
- KL 惩罚项加到奖励中：$r_\text{eff} = r_\phi - \beta \cdot \text{KL}(\pi_\theta || \pi_\text{ref})$，限制策略偏移范围（3分）

---

### Q7【简答，10分】
PPO 为什么可以用同一批数据更新多次（而 REINFORCE 不行）？

**答案（评分标准）**：
- 核心：PPO 用重要性采样（Importance Sampling），通过概率比率 $r_t = \pi_\theta/\pi_{\theta_\text{old}}$ 修正 off-policy 偏差（4分）
- REINFORCE 是 on-policy：梯度估计假设数据来自当前策略，多次更新后当前策略已变，数据变旧失效（3分）
- PPO 用 Clip 进一步防止更新过大（3分）

---

### Q8【代码理解，6分】
下面的代码有一个 bug，找出并说明应如何修复：

```python
def ppo_loss(log_probs_new, log_probs_old, advantages, eps=0.2):
    ratio = torch.exp(log_probs_new - log_probs_old)  # 未 detach
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages
    return -torch.mean(torch.min(surr1, surr2))
```

**答案**：`log_probs_old` 应该调用 `.detach()`，改为 `torch.exp(log_probs_new - log_probs_old.detach())`。  
**评分标准**：
- 找出 bug（3分）
- 解释原因：`log_probs_old` 是旧策略采集数据时的固定参考值，不应参与当前策略的梯度计算（3分）

---

### Q9【代码理解，6分】
以下 `compute_returns` 函数中，`returns.std() + 1e-8` 这行在做什么？删掉 `+ 1e-8` 会有什么风险？

```python
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns
```

**答案**：  
- `+ 1e-8` 是数值稳定性技巧（3分）  
- 删掉后，若所有奖励完全相同（std=0），会触发除以零（ZeroDivisionError 或 NaN），导致训练崩溃（3分）

---

### Q10【代码理解，6分】
在 GAE 实现中：`gae = delta + gamma * lam * gae`（先算后更新）。如果改为正向遍历（从 $t=0$ 开始），代码还能正确实现 GAE 吗？为什么？

**答案**：不能。GAE 定义为 $\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$，需要知道未来步的优势才能计算当前步。正向遍历时 $\hat{A}_{t+1}$ 还未知，因此必须从末尾向前计算（4分）。正确的递推公式 $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$ 要求先计算 $\hat{A}_{t+1}$（2分）。
