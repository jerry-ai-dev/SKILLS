# Lesson 11: DPO 直接偏好优化（与 IPO / KTO / ORPO / SimPO 变体）

## 教学目标
- 理解 DPO 为什么能"绕开 Reward Model + PPO"直接用偏好数据训练
- 从 Bradley-Terry + 带 KL 约束的 RLHF 目标推导出 DPO 闭式解
- 写出并解读 DPO loss，掌握 `β`、`ref_model`、chosen/rejected 三件套
- 能对比 DPO 与 PPO / GRPO 的取舍，了解 IPO / KTO / ORPO / SimPO 等主流变体

---

## 讲解要点

### 1. 为什么需要 DPO：标准 RLHF 太重了

回忆 Lesson 5 的标准 RLHF：

```
SFT  →  Reward Model 训练  →  PPO / GRPO 在线 RL
```

工程上要做到三件事：
- 训练并维护一个独立的 **Reward Model**（一个 LLM 级别的网络）
- 在线**采样** + 在线打分（要么靠 RM forward，要么靠规则函数）
- 跑 PPO/GRPO，至少要 Actor + Ref + （Critic 或 多次采样）三份模型常驻显存

**核心观察**（Rafailov et al., 2023）：如果 reward 函数是从偏好数据按 Bradley-Terry 拟合出来的，而最终 RL 目标又是带 KL 约束的最大化 reward，那这两步**可以解析地合并成一个监督式 loss**——不再需要独立的 RM，也不再需要在线采样。

这就是 **DPO（Direct Preference Optimization）**。

> 一句话：**DPO 把"训练 RM + 跑 PPO"两个阶段，压缩成一次像 SFT 一样的离线监督训练。**

---

### 2. 推导：从 RLHF 目标到 DPO 闭式解

#### Step 1：RLHF 的带 KL 约束目标

$$
\max_{\pi_\theta}\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)} \big[r(x,y)\big]
\;-\; \beta \cdot \mathrm{KL}\!\left(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\right)
$$

这是 Lesson 5 的核心公式：最大化期望奖励，同时不让策略偏离参考模型太远。

#### Step 2：这个目标的最优策略有闭式解

可以证明（拉格朗日 + 一阶条件，留作选做练习）：

$$
\pi_r^*(y|x) \;=\; \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$ 是配分函数。

**反解 reward**：

$$
r(x,y) \;=\; \beta \log \frac{\pi_r^*(y|x)}{\pi_{\text{ref}}(y|x)} \;+\; \beta \log Z(x)
$$

> **人话翻译**：reward 不是凭空定义的——它就等于"最优策略 / 参考策略 的对数比"再乘 β（加上一个跟 y 无关的常数 $\log Z(x)$）。

#### Step 3：代入 Bradley-Terry，让 $Z(x)$ 消掉

Lesson 4 的 Bradley-Terry 偏好模型：

$$
P(y_w \succ y_l \mid x) \;=\; \sigma\big(r(x,y_w) - r(x,y_l)\big)
$$

把 Step 2 的 reward 表达式代进去，**$\beta \log Z(x)$ 在 $y_w$ 和 $y_l$ 中是同一个，相减就抵消掉了**：

$$
P(y_w \succ y_l \mid x) \;=\; \sigma\!\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

> **人话**：原来 RM 要先学 $r(x,y)$ 再去打分，现在我们直接用 **策略和参考策略的对数比** 当 reward，跳过 RM 这一步。

#### Step 4：写成 DPO loss

对偏好数据集做最大似然，即取负对数：

$$
\boxed{\;
\mathcal{L}_{\text{DPO}}(\theta) \;=\; - \mathbb{E}_{(x, y_w, y_l)}\;\log \sigma\!\left[\,\beta\,\big(\log \tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big)\,\right]
\;}
$$

记号简化：定义 **隐式奖励** $\hat r_\theta(x,y) := \beta \log \tfrac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，则

$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}\big[\log\sigma(\hat r_\theta(x,y_w) - \hat r_\theta(x,y_l))\big]
$$

形式**和 RM 训练时的 Bradley-Terry loss 一模一样**——只不过把"显式 reward head"换成了"对数策略比"。

---

### 3. DPO 在做什么？梯度直觉

对 DPO loss 求梯度（不展开推导，结论很关键）：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}}
\;=\; - \beta \cdot \mathbb{E}\big[\, \sigma(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)) \cdot \big(\nabla\log\pi_\theta(y_w|x) - \nabla\log\pi_\theta(y_l|x)\big)\,\big]
$$

**这告诉我们三件事**：

1. **方向**：提高 $\log\pi_\theta(y_w|x)$、压低 $\log\pi_\theta(y_l|x)$——和 RLHF 想做的事完全一致。
2. **自带样本权重**：$\sigma(\hat r(y_l) - \hat r(y_w))$ 是"模型当前给错答案打的隐式分 - 给对答案的分"的 sigmoid。**模型已经分对了的样本权重小，分错得离谱的样本权重大**，相当于自动 hard example mining。
3. **`ref_model` 是锚**：所有公式里都有 $\pi_{\text{ref}}$，它的角色就是 Lesson 5 那个 KL 锚——防止策略飘走。

---

### 4. DPO 训练所需的"三件套"

| 物件 | 角色 | 备注 |
|---|---|---|
| 偏好数据集 $(x, y_w, y_l)$ | 监督信号 | 和训练 RM 用的数据**完全一样** |
| 当前策略 $\pi_\theta$ | 要训练的模型 | 通常从 SFT checkpoint 初始化 |
| 参考策略 $\pi_{\text{ref}}$ | 冻结，提供 KL 锚 | 一般就是 SFT 模型本身 |

工程上 forward 一次要算 **4 份 log_probs**：$\pi_\theta(y_w), \pi_\theta(y_l), \pi_{\text{ref}}(y_w), \pi_{\text{ref}}(y_l)$——所以显存 ≈ 2 份模型大小（policy + ref，每份做两次 forward），**比 PPO 省，但比纯 SFT 贵**。

---

### 5. 超参 β 的取舍

| β 取值 | 直觉 | 风险 |
|---|---|---|
| β 小（如 0.01） | 几乎不管 ref，激进拉开 chosen / rejected 的差距 | 容易过拟合偏好集 / 退化 / "样式塌缩" |
| β 中（如 0.1 ~ 0.5） | TRL 默认 0.1，多数任务 sweet spot | — |
| β 大（如 1.0+） | KL 约束很强，几乎不偏离 SFT | 偏好信号传不进去，几乎没效果 |

**常见经验**：先用 β=0.1 跑一遍，看 chosen / rejected 的隐式 reward 差距曲线是否在合理区间（既不 0 也不爆炸）再调。

---

### 6. DPO vs PPO/GRPO 横向对比

| 维度 | PPO | GRPO | **DPO** |
|---|---|---|---|
| 是否需要独立 RM | ✅ 要 | ✅ 要（或规则）| ❌ **不需要** |
| 是否在线采样 | ✅ 要 | ✅ 要 | ❌ **离线，偏好数据即可** |
| 是否需要 Critic | ✅ 要 | ❌ 不要 | ❌ 不要 |
| 显存峰值 | ~4× 模型 | ~2-3× 模型 | ~2× 模型 |
| 训练像什么 | 在线 RL | 在线 RL | **更像加权的 SFT** |
| 推理能力涌现（数学/代码） | 强 | **强**（R1 走通） | 弱（偏好集多为对话风格） |
| 适合场景 | 通用对齐 | 可验证奖励（数学/代码） | **对话偏好对齐 / 风格 / 安全** |
| 工程复杂度 | 高 | 中 | **低** |

**抓重点**：
- **可验证奖励** + 想涌现推理 → **GRPO**（R1 路线）
- **只有偏好对、想稳定低成本对齐** → **DPO**
- 两者**不是替代关系**，工业管线里经常 **SFT → DPO → GRPO** 串联。

---

### 7. DPO 的常见局限 & 主流变体

#### 局限
1. **离线偏好数据分布有限**：模型见不到自己采样的新轨迹，难涌现新行为
2. **可能 "样式塌缩"**：β 太小、训练太久，模型把 chosen 风格无限放大，整体多样性下降
3. **依赖偏好对质量**：标注噪声直接进梯度，没有 RM 那一层平滑

#### 变体一览（理解层面，不展开推导）

| 名称 | 一句话差别 | 何时用 |
|---|---|---|
| **IPO**（Identity Preference Optimization） | 把 $\log\sigma$ 换成平方项，抑制 DPO 的过拟合 | 偏好数据噪声大或标注模糊 |
| **KTO**（Kahneman-Tversky） | 不需要成对 (chosen, rejected)，单条标"好/不好"也能训 | 偏好数据只有单边标签 |
| **ORPO** | 把 SFT loss 和偏好 loss 合并，**不需要 ref_model** | 想再省一份模型显存 |
| **SimPO** | 用平均 log-prob 当隐式 reward，**不需要 ref_model** + 加长度归一 | 想减显存、缓解长度偏置 |
| **CPO / RPO / 其它** | 围绕"如何更稳/更省"的工程改进 | 视具体场景挑 |

> **共同主题**：这一票算法都在和 DPO 抢同一个生态位——**只用离线偏好数据做对齐**，差别在于显存、稳定性、是否需要 ref。

---

### 8. 在 DeepSeek R1 / 你的项目里 DPO 在哪？

- **DeepSeek R1**：主线是 GRPO（因为奖励可规则验证）。但在论文里第二轮 SFT 之后的 helpfulness/harmlessness 对齐阶段，DPO/偏好对齐类方法是工业界标准做法之一。
- **你的项目**：如果你想做数学推理 → 走 GRPO（Lesson 6 路线）。如果你想做对话风格 / 安全 / 角色扮演对齐 → DPO 是入门门槛最低、最便宜的选择。

---

## 代码示例

```python
import torch
import torch.nn.functional as F

# ===== DPO Loss 核心实现（mini 版本，约 30 行）=====

def get_logprob_of_sequence(model, input_ids, response_mask):
    """
    返回每条样本"回答部分"的总 log 概率（标量 per sample）。
    input_ids:     [B, T]
    response_mask: [B, T]，1 = 回答 token，0 = prompt / padding
    """
    logits = model(input_ids).logits[:, :-1, :]            # [B, T-1, V]
    targets = input_ids[:, 1:]                              # [B, T-1]
    mask = response_mask[:, 1:]                             # [B, T-1]
    log_probs = F.log_softmax(logits, dim=-1)               # [B, T-1, V]
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    return (token_logp * mask).sum(dim=-1)                  # [B]


def dpo_loss(
    policy_logp_chosen:    torch.Tensor,   # [B] log π_θ(y_w | x)
    policy_logp_rejected:  torch.Tensor,   # [B] log π_θ(y_l | x)
    ref_logp_chosen:       torch.Tensor,   # [B] log π_ref(y_w | x)  (no grad)
    ref_logp_rejected:     torch.Tensor,   # [B] log π_ref(y_l | x)  (no grad)
    beta: float = 0.1,
):
    """
    DPO loss + 一些有用的诊断量
    """
    # 隐式 reward：β · log( π_θ / π_ref )
    chosen_implicit_reward   = beta * (policy_logp_chosen   - ref_logp_chosen.detach())
    rejected_implicit_reward = beta * (policy_logp_rejected - ref_logp_rejected.detach())

    # logits = β · ( Δlog π_θ - Δlog π_ref )
    logits = chosen_implicit_reward - rejected_implicit_reward

    # 负 logsigmoid（数值稳定）= -log σ(logits)
    loss = -F.logsigmoid(logits).mean()

    # 诊断：reward margin、隐式 reward 准确率（越靠近 1 越好）
    metrics = {
        "loss":               loss.item(),
        "reward_margin":      (chosen_implicit_reward - rejected_implicit_reward).mean().item(),
        "reward_acc":         (chosen_implicit_reward > rejected_implicit_reward).float().mean().item(),
        "chosen_reward_mean": chosen_implicit_reward.mean().item(),
        "rejected_reward_mean": rejected_implicit_reward.mean().item(),
    }
    return loss, metrics


# ===== Demo（用伪造的 log_prob 而不是真的跑模型，CPU 几毫秒）=====
print("==== DPO Loss 演示 ====\n")

torch.manual_seed(0)
B = 4
# 模拟：policy 已经学得不错（chosen log_prob 更高）
policy_logp_chosen   = torch.tensor([-12.0, -15.0, -10.0, -20.0])
policy_logp_rejected = torch.tensor([-18.0, -14.0, -16.0, -19.0])
# ref 模型是 SFT，两边差不多
ref_logp_chosen      = torch.tensor([-14.0, -14.5, -13.0, -19.5])
ref_logp_rejected    = torch.tensor([-15.0, -14.0, -14.5, -19.0])

loss, metrics = dpo_loss(
    policy_logp_chosen, policy_logp_rejected,
    ref_logp_chosen, ref_logp_rejected,
    beta=0.1,
)
for k, v in metrics.items():
    print(f"  {k:24s}: {v:+.4f}")

print("\n==== 关键观察 ====")
print("1. reward_margin > 0：策略相对 ref 更偏好 chosen，方向正确")
print("2. reward_acc ≈ 1.0：所有样本上隐式 reward 都把 chosen 排在前面")
print("3. 训练目标就是把 reward_margin 拉大、把 loss 压低")
```

运行后应该能看到 `reward_margin > 0` 且 `reward_acc = 1.0`。**实战训练时盯这两个指标比盯 loss 更有用**——loss 数值意义不大，但 margin 是否在涨、acc 是否接近 1 直接反映训练是否正常。

---

## 测验题

**Q1（选择）** 关于 DPO，下列说法**错误**的是：
- A. DPO 不需要单独训练 Reward Model
- B. DPO 不需要在线采样，可以用离线偏好数据 (x, y_w, y_l) 直接训练
- C. DPO 不需要参考模型 $\pi_{\text{ref}}$，所以比 PPO 显存少
- D. DPO 的 loss 形式上和 Bradley-Terry RM loss 几乎一样，只是把显式 reward 换成对数策略比

**答案**：C。DPO **仍然需要 $\pi_{\text{ref}}$** 作为 KL 锚——它的显存里同时有 policy 和 ref 两个模型（约 2× 模型大小）。"不需要 ref" 的是 ORPO / SimPO 等变体。

---

**Q2（推导）** 写出 DPO loss 的完整公式，并解释为什么参考模型的 log_prob 必须 `detach`（即不参与梯度）。

**答案要点**：
$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}\,\log\sigma\!\left[\beta\big(\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big)\right]
$$
- $\pi_{\text{ref}}$ 是冻结的 SFT 模型，**只用来定义"相对偏离量"**，本身不更新参数。
- 若不 detach，反向传播会试图通过 $\pi_{\text{ref}}$ 的 log_prob 更新参数，要么报错（参数没 requires_grad），要么逻辑混乱。

---

**Q3（代码）** 下列 DPO loss 实现有 **3 个 bug**，全部找出：

```python
def dpo_loss(p_w, p_l, r_w, r_l, beta):
    logits = beta * (p_w / p_l) - beta * (r_w / r_l)   # Bug 1
    loss = F.sigmoid(logits).mean()                     # Bug 2
    return loss                                         # Bug 3 (ref 没 detach)
```

**答案**：
1. **概率比要在 log 空间用减法**：`logits = beta * ((p_w - p_l) - (r_w - r_l).detach())`（这里 `p_w/p_l` 是 log_prob，直接做除法会得到没物理意义的数值）
2. **应该是 `-F.logsigmoid` 不是 `F.sigmoid`**：`loss = -F.logsigmoid(logits).mean()`，且 logsigmoid 数值稳定
3. **`r_w, r_l` 必须 `.detach()`**：上面已合并修复

---

**Q4（概念）** 工程上看，DPO 和 GRPO 各自最擅长什么场景？为什么 DeepSeek R1 选 GRPO 而不是 DPO？

**答案要点**：
- DPO 擅长**离线偏好对齐**（对话风格、安全、角色），不需要在线采样，工程便宜，但只能学到偏好集里的模式。
- GRPO 擅长**可验证奖励**（数学对错、代码能不能运行）下的**在线** RL，能让模型自己采样新轨迹、涌现新策略。
- R1 的目标是涌现长链推理能力，奖励天然可以规则验证（答案对 vs 错），需要**在线探索 + 自我反思**——DPO 是离线监督，根本无法产生"Aha moment"这种新行为。所以选 GRPO。

---

## 课后练习（选做）

1. **推导**：从 RLHF 带 KL 约束目标出发，用拉格朗日法证明最优策略
   $\pi_r^*(y|x) = \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta) / Z(x)$。
2. **代码**：把 `dpo_loss` 改造成 **IPO**——把 `-log σ(·)` 换成 $(\cdot - 1/(2\beta))^2$，看看在带噪偏好数据上是否更稳。
3. **思考**：如果偏好集里 80% 的样本 chosen 都比 rejected **长很多**，DPO 训完会发生什么？这就是 **"长度偏置"**——SimPO 为什么要做长度归一？
4. **对比**：画一张表，把 DPO 和 ORPO 在"是否需要 ref"、"是否合并 SFT loss"、"显存开销"三个维度上的差别写清楚。
