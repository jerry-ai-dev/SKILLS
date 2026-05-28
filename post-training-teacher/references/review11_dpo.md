# 📖 Lesson 11 复习八股文：DPO 直接偏好优化

> 用最精炼的概念 + 最生动的类比，帮你 5 分钟回忆本课所有核心知识点。

---

## 一、核心概念速查表

| 概念 | 一句话定义 | 生活类比 |
|------|-----------|---------|
| DPO | 用偏好数据直接训练策略，跳过 RM + PPO 两阶段 | 把"先请人打分、再按分调整"两步合成"看到优劣对就直接学" |
| 隐式 reward $\hat r_\theta$ | $\beta \log(\pi_\theta / \pi_{\text{ref}})$，由策略和参考策略自动构造 | 不请专家打分了，直接看"你现在比基线好多少"作为分数 |
| 参考模型 $\pi_{\text{ref}}$ | 冻结的 SFT 模型，提供 KL 锚 | 体能训练里的"原地基线"，所有改变都相对它衡量 |
| β | KL 约束强度 | 训练时的"安全绳"——小=激进、大=保守 |
| chosen / rejected | 偏好对里的好回答 / 差回答 | 双盲品酒里的优胜样和淘汰样 |

---

## 二、关键公式卡片

### DPO Loss
$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}_{(x,y_w,y_l)} \log\sigma\!\left[\beta\big(\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big)\right]
$$
> **人话**：用"策略相对 ref 的对数比"代替显式 reward，套进 Bradley-Terry 偏好 loss 就完事。

### 隐式 reward
$$
\hat r_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$
> **人话**：你的策略给这条回答的概率，相对参考模型涨了多少，就当作 reward。

### 闭式最优策略（推导起点）
$$
\pi_r^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp\!\big(r(x,y)/\beta\big)
$$
> **人话**：RLHF 最优解 = 参考模型上叠加一个"reward 越大概率越高"的指数权重。

---

## 三、生动类比：双盲品酒大赛

想象你是一家酒厂的酿酒师（**policy** $\pi_\theta$），目标是学会酿出大众更喜欢的酒。

- **PPO/GRPO 的做法**：先雇一群品酒师当评委（**Reward Model**），让他们每瓶都打分；然后你每酿一批就送过去打分，再根据分数调整配方。**评委要请、酒要现场酿现场送、流程很重**。
- **DPO 的做法**：评委直接给你一堆"A 比 B 好喝"的成对意见（偏好对），你**自己悟**——"那我以后多走 A 这个方向"。不需要请评委，不需要现场打分，**翻翻历史记录就能学**。

> **核心洞察**：RLHF 想做的事其实是"提高好回答的概率、压低差回答的概率"。DPO 发现这事根本不用绕一圈 reward model——**直接对着偏好对调概率就行了**。

---

## 四、DPO 训练为什么"看起来像 SFT"

| 维度 | SFT | DPO | PPO |
|---|---|---|---|
| 数据 | (prompt, response) | (prompt, chosen, rejected) | prompt + 在线采样 |
| Loss | Cross-Entropy | -log σ(β·Δlog ratio) | Clip + KL |
| 在线采样 | ❌ | ❌ | ✅ |
| 需要 RM | ❌ | ❌ | ✅ |
| 需要 ref_model | ❌ | ✅ | ✅ |
| 工程像什么 | 监督学习 | **加权监督学习** | 在线 RL |

> **记忆钩子**：DPO 是"**带 ref 锚的、对偏好对加权的 SFT**"。

---

## 五、DPO Loss 的四个组件拆解

```
DPO Loss = -log σ(  logits  )

logits = β · ( Δ_policy − Δ_ref )
              ↑           ↑
              ↑           └ log π_ref(y_w) − log π_ref(y_l)   (detach!)
              └ log π_θ(y_w) − log π_θ(y_l)
```

- **① β**：KL 强度，默认 0.1
- **② Δ_policy**：当前策略对 chosen / rejected 的相对偏好
- **③ Δ_ref**：参考模型的同款相对偏好（用来做基线扣除）
- **④ -log σ(·)**：让 logits 越大越好——即让"策略偏好 chosen 的程度"超过"ref 偏好 chosen 的程度"

---

## 六、DPO vs PPO vs GRPO 终极对比

| 维度 | PPO | GRPO | **DPO** |
|---|---|---|---|
| 独立 RM | ✅ | ✅ / 规则 | ❌ |
| 在线采样 | ✅ | ✅ | ❌ 离线 |
| Critic | ✅ | ❌ | ❌ |
| ref_model | ✅ | ✅ | ✅ |
| 显存峰值 | ~4× | ~2-3× | ~2× |
| 是否能涌现新行为 | 强 | **强**（R1）| 弱 |
| 工程复杂度 | 高 | 中 | **低** |
| 工业典型用途 | 通用对齐 | 数学/代码推理 | 对话风格/安全/角色 |

---

## 七、DPO 的局限与变体一览

| 名称 | 核心改动 | 解决什么问题 |
|---|---|---|
| **IPO** | $-\log\sigma$ → 平方项 | DPO 在带噪偏好上易过拟合 |
| **KTO** | 用 prospect theory，单边标签也能训 | 没有成对数据，只有"好/差"判断 |
| **ORPO** | 合并 SFT + 偏好 loss，**不要 ref** | 想再省一份模型显存 |
| **SimPO** | 平均 log-prob 当 reward，**不要 ref** + 长度归一 | 显存 / 长度偏置 |

> **共同主题**：抢同一个生态位——**只用离线偏好对齐**，差别在显存 / 稳定性 / 是否要 ref。

---

## 八、训练时盯哪些指标

| 指标 | 含义 | 健康范围 |
|---|---|---|
| `reward_margin` | 隐式 reward(chosen) − rejected 的均值 | 稳定上涨（≥ 0.5 比较好） |
| `reward_acc` | chosen > rejected 的比例 | 接近 1.0 |
| `chosen_rewards` | β · Δlog ratio of chosen | 缓慢上涨 |
| `rejected_rewards` | β · Δlog ratio of rejected | 缓慢下降 |
| `loss` | DPO loss | 缓慢下降，但绝对值意义不大 |

**经验**：loss 数值意义不大，但 **margin 涨 + acc → 1** 才是训练正常的真信号。

---

## 九、易混淆点 & 常见误区

1. **DPO ≠ 不需要 ref**：DPO 仍然要加载参考模型，显存 ~2×。"不要 ref" 的是 ORPO / SimPO。
2. **DPO 的 reward 是"隐式"**：从来没有显式 RM，那个 $\hat r_\theta = \beta \log(\pi/\pi_{\text{ref}})$ 是事后定义出来的。
3. **`ref_logp` 必须 detach**：否则梯度会试图反传到冻结模型。
4. **β 不是学习率**：β 是 KL 约束强度，调它影响"模型敢偏离 ref 多远"。
5. **DPO 不能凭空涌现数学推理**：它只能强化偏好集里已有的模式。要想涌现长链反思，得 GRPO + 可验证奖励。

---

## 十、记忆口诀

> **"双盲品酒比好坏，对数策略当分数；锚住 SFT 不漂移，β 大小定胆量"**
> - 双盲品酒 = 离线偏好对 (chosen, rejected)
> - 对数策略当分数 = 隐式 reward $= \beta \log(\pi/\pi_{\text{ref}})$
> - 锚住 SFT = 参考模型 $\pi_{\text{ref}}$ 提供 KL 锚
> - β 定胆量 = β 小=激进、β 大=保守

---

## 十一、自测题（快速检验）

1. DPO 的核心 loss 公式是什么？里面的"隐式 reward"怎么定义？
2. 为什么 DPO 仍然需要 $\pi_{\text{ref}}$？它和 PPO 里的 ref_model 角色一样吗？
3. 工业管线里经常出现 "SFT → DPO → GRPO" 这种串联，每一步分别解决什么问题？
4. DPO 训练时为什么要盯 `reward_margin` 和 `reward_acc` 而不是只看 loss？
5. 如果你的偏好集 80% 的 chosen 比 rejected 长很多，DPO 会出现什么问题？哪个变体专门修这个？

> 如果以上问题都能不翻笔记快速回答，恭喜你——Lesson 11 已稳！
