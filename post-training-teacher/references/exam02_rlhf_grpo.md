# Exam 2: RLHF 完整流程阶段考试

## 考试说明
- **范围**：Lesson 4-7（Reward Model、RLHF、GRPO、SFT）
- **题数**：10 题，满分 100 分
- **分布**：选择题 3 道（每题 8 分）+ 推导/简答题 3 道（每题 12 分）+ 代码题 4 道（每题 8 分）
- **规则**：逐题作答，建议先不看笔记

---

## 题目

### Q1【选择，8分】
Bradley-Terry Reward Model 的训练 Loss $\mathcal{L} = -\log\sigma(r_w - r_l)$ 中，$r_w$ 和 $r_l$ 分别代表：

A. 训练集和测试集的奖励  
B. 胜者(chosen)和败者(rejected)回答的奖励分数  
C. 当前策略和参考策略的 log 概率  
D. 优势函数的正部分和负部分  

**答案**：B  
**解析**：Bradley-Terry 模型假设人类选择胜者的概率 $= \sigma(r_w - r_l)$，最大化这个概率等价于最小化 $-\log\sigma(r_w - r_l)$

---

### Q2【选择，8分】
GRPO 和 PPO 的最主要工程区别是：

A. GRPO 使用不同的 clip 函数  
B. GRPO 不需要 Critic（价值网络），用组内相对奖励代替优势估计  
C. GRPO 不使用重要性采样  
D. GRPO 的学习率更低  

**答案**：B  
**解析**：GRPO 去掉了与 Actor 同等规模的 Critic 网络，改用同一组生成的奖励均值做 baseline，节省约一半显存

---

### Q3【选择，8分】
SFT 训练中，对 Prompt 对应的 token 位置设置 `label = -100` 的目的是：

A. 标记这些位置是重要的，需要重点训练  
B. 告诉 `CrossEntropyLoss` 忽略这些位置，不计算 loss，不传梯度  
C. 把这些 token 从输入中删除  
D. 降低这些位置的学习率  

**答案**：B  
**解析**：`CrossEntropyLoss(ignore_index=-100)` 跳过标签为 -100 的位置，即只在 Assistant 回复部分计算损失

---

### Q4【推导，12分】
写出 GRPO Loss 的完整公式并解释每个符号的含义。特别说明：为什么优势 $\hat{A}_i$ 对整条回答的所有 token 使用相同的值？

**答案（评分标准）**：
$$\mathcal{L}_\text{GRPO} = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\left[\min(r_{t,i}\hat{A}_i, \text{clip}(r_{t,i},1-\epsilon,1+\epsilon)\hat{A}_i) - \beta\cdot\text{KL}_t\right]$$

- 写出完整公式（4分）
- 解释 $r_{t,i}$：第 i 条回答第 t 个 token 的新旧策略概率比（2分）
- 解释 $\hat{A}_i = (r_i - \text{mean}) / \text{std}$：组内归一化优势（2分）
- 解释为什么整条共用：GRPO 对整条回答只有一个整体奖励（如"答对"），无法区分哪个 token 更重要，所以整条共用同一个优势（4分）

---

### Q5【简答，12分】
描述 RLHF 完整训练的三个阶段，以及每个阶段的输入数据、损失函数、输出是什么？

**答案（评分标准）**：

**阶段1 SFT**（4分）：
- 输入：(prompt, high-quality response) 对，人工标注
- 损失：Cross-Entropy Loss（只在 response 部分）
- 输出：SFT 模型 $\pi_\text{SFT}$

**阶段2 RM 训练**（4分）：
- 输入：(prompt, chosen_response, rejected_response) 三元组
- 损失：$-\log\sigma(r_w - r_l)$（Bradley-Terry preference loss）
- 输出：奖励模型 $r_\phi$

**阶段3 PPO/GRPO RL**（4分）：
- 输入：无标注 prompt，用 RM 打分
- 损失：PPO/GRPO Clip Loss + KL 惩罚
- 输出：对齐后的策略 $\pi_\theta^*$

---

### Q6【简答，12分】
什么是序列 Packing？为什么它能提高 GPU 利用率？举一个具体例子说明 packing 前后的区别。

**答案（评分标准）**：
- 定义（3分）：将多条短序列拼接成一条达到 max_seq_length 的长序列，用 attention mask 确保不同条数据间互不 attend
- 提高效率原因（3分）：减少 padding，使每个 batch 的有效 token 比例接近 100%，充分利用 GPU 并行计算
- 具体例子（6分）：
  - 不 packing：3条长200/100/50 token的序列，pad到2048，有效token比例=(200+100+50)/(2048×3)≈5.7%
  - packing：三条拼接=350 token，填入2048长度，有效比例≈17%，或更多条可拼到90%+

---

### Q7【代码，8分】
下面的 GRPO advantage 计算有误，找出并修复：

```python
def grpo_advantage(rewards):
    # rewards: [G] 同一 prompt 下 G 条回答的奖励
    return (rewards - rewards.mean()) / rewards.std()
```

**答案**：缺少数值稳定性处理，当所有奖励相同时 `std()=0` 导致除以零（NaN）。  
修复：`return (rewards - rewards.mean()) / (rewards.std() + 1e-8)`  
**评分标准**：找出 bug 4分，写出正确代码 4分

---

### Q8【代码，8分】
下面是一段 SFT 数据预处理代码，有一个严重错误：

```python
def prepare_sft_labels(input_ids, prompt_len):
    # input_ids: [seq_len]
    labels = input_ids.clone()
    # 让模型学习回答部分
    labels[:prompt_len] = 0   # 将 prompt 部分设为 0
    return labels
```

问题在哪里？正确的做法是什么？

**答案**：应该将 prompt 部分设为 `-100`，而不是 `0`。设为 `0` 会让 CrossEntropyLoss 把 token id=0（通常是 `<pad>` 或某个实际 token）也计入 loss，模型会被错误地训练成在 prompt 位置预测 id=0 的 token（4分）。正确做法：`labels[:prompt_len] = -100`（4分）

---

### Q9【代码，8分】
下面是 Reward Model 的前向传播，请解释 `squeeze(-1)` 的作用，并说明为什么 RM 的输出是标量而不是 token-level 的分布：

```python
class RewardModel(nn.Module):
    def forward(self, x):
        features = self.backbone(x)       # [batch, seq_len, hidden]
        last_hidden = features[:, -1, :]  # [batch, hidden]
        reward = self.reward_head(last_hidden).squeeze(-1)  # [batch]
        return reward
```

**答案**：  
- `squeeze(-1)` 的作用（4分）：`reward_head` 是 Linear(hidden, 1)，输出 shape [batch, 1]，`squeeze(-1)` 去掉最后的维度 1，变为 [batch]，方便后续与其他标量运算
- 为什么是标量（4分）：RM 需要对**整条回答**打一个质量分，用于训练时的比较和 RL 阶段的奖励信号；取最后一个 token 的 hidden state 是因为 LLM 用 causal attention，最后一个 token 已汇聚了全部上下文信息

---

### Q10【代码，8分】
以下 `preference_loss` 代码中，如果 `reward_chosen` 和 `reward_rejected` 的顺序传反了（即把差的回答传给 chosen，好的回答传给 rejected），训练结果会怎样？写出推导过程。

```python
def preference_loss(reward_chosen, reward_rejected):
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()
```

**答案**：若传反了，loss = $-\log\sigma(r_\text{bad} - r_\text{good})$。训练时模型会朝着让 $r_\text{bad} > r_\text{good}$ 的方向更新，即学习给差回答更高分（4分）。这是一个严重的数据标注方向错误，模型会"学反"，对所有问题给出更差的回答评估，实际效果比不训练还差（4分）。
