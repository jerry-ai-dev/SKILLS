# Lesson 2: TRL GRPOTrainer 源码精读

## 学习目标
- 理解 GRPOTrainer 的整体训练循环
- 看懂采样、优势计算、策略更新、KL 惩罚四个核心模块的代码
- 能将阶段二的公式与真实代码一一对应

## 知识小节

### 小节 1：GRPOTrainer 整体架构

GRPOTrainer 的训练循环（每一步）：

```
1. 从数据集取一批 prompt
2. 用当前策略生成 G 条候选回答（采样）
3. 用奖励函数给每条回答打分
4. 计算组内优势（z-score 标准化）
5. 计算 PPO-Clip loss + KL 惩罚
6. 反向传播更新参数
```

```python
# trl/trainer/grpo_trainer.py（核心训练循环简化版）

class GRPOTrainer(Trainer):
    def __init__(self, model, reward_funcs, args, train_dataset, ...):
        self.model = model                    # 当前策略 π_θ
        self.ref_model = create_reference_model(model)  # 参考模型 π_ref（冻结）
        self.reward_funcs = reward_funcs      # 奖励函数列表
        ...
    
    def _generate_and_score(self, prompts):
        """Step 2-3: 生成候选回答并打分"""
        # 对每个 prompt 生成 G 条回答
        all_completions = self.model.generate(
            prompts,
            num_return_sequences=self.args.num_generations,  # G
            temperature=self.args.temperature,
            max_new_tokens=self.args.max_completion_length,
        )
        
        # 用奖励函数打分
        rewards = []
        for reward_func in self.reward_funcs:
            rewards.append(reward_func(prompts, all_completions))
        
        return all_completions, rewards
    
    def _compute_advantages(self, rewards):
        """Step 4: 组内 z-score 标准化"""
        # rewards shape: [batch_size, G]
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-8)
        return advantages
    
    def _compute_loss(self, model_output, ref_output, advantages, mask):
        """Step 5: PPO-Clip loss + KL 惩罚"""
        # 计算 ratio = π_new / π_old
        log_ratio = model_output.log_probs - model_output.old_log_probs
        ratio = torch.exp(log_ratio)
        
        # PPO-Clip
        clipped_ratio = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
        
        # KL 惩罚
        kl = model_output.log_probs - ref_output.log_probs.detach()
        kl_loss = self.args.beta * kl
        
        # 总 loss（只对有效 token 计算）
        loss = ((policy_loss + kl_loss) * mask).sum() / mask.sum()
        return loss
```

### 小节 2：采样模块详解

```python
# 采样的关键参数
generation_config = {
    "num_return_sequences": G,        # 每个 prompt 生成 G 条回答
    "temperature": 0.7,                # 控制生成多样性
    "top_p": 0.95,                     # nucleus sampling
    "max_new_tokens": 512,             # 最大生成长度
    "do_sample": True,                 # 必须开启随机采样（不能 greedy）
}
```

**为什么不用 greedy（贪心）？**
- GRPO 需要**多样性**——同一个 prompt 的 G 条回答要有差异
- 如果 greedy，G 条回答全一样 → 奖励全一样 → 优势全为 0 → 学不到东西
- temperature 越高越多样，但太高会导致答案质量下降

### 小节 3：优势计算 — z-score 标准化

```python
def compute_advantages(rewards):
    """
    rewards: Tensor of shape [batch_size, G]
    对应阶段二公式：A_hat_i = (r_i - mean) / std
    """
    mean = rewards.mean(dim=-1, keepdim=True)   # 组内均值
    std = rewards.std(dim=-1, keepdim=True)      # 组内标准差
    advantages = (rewards - mean) / (std + 1e-8) # z-score + 防除零
    return advantages

# 示例
rewards = torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # 4条回答，2对2错
advantages = compute_advantages(rewards)
# mean=0.5, std=0.5
# advantages = [[+1.0, -1.0, +1.0, -1.0]]
# 对的鼓励(+)，错的打压(-)
```

### 小节 4：策略更新 — ratio 和 PPO-Clip

```python
def compute_policy_loss(new_log_probs, old_log_probs, advantages, epsilon=0.2):
    """
    对应阶段二公式：
    L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    
    对应阶段二 Exam 3 Q10 的修正版：
    - ratio = exp(log_new - log_old)（不是相除！）
    - 加负号变成 loss
    """
    # 计算 ratio（对应 Q10 的 Bug 1 修正）
    log_ratio = new_log_probs - old_log_probs.detach()
    ratio = torch.exp(log_ratio)
    
    # PPO-Clip
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    # 取 min 保守更新（对应阶段二 Lesson 3 PPO 理论）
    loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    )
    return loss
```

### 小节 5：KL 惩罚 — per-token KL

```python
def compute_kl_penalty(log_probs_policy, log_probs_ref, attention_mask, beta=0.1):
    """
    对应阶段二 Exam 3 Q12 的三行代码
    对应阶段二 Q7 的 β 参数分析
    """
    # Line A: 逐 token 的 KL 贡献
    token_kl = log_probs_policy - log_probs_ref.detach()
    
    # Line B: 过滤 padding
    token_kl = token_kl * attention_mask
    
    # Line C: 对有效 token 求平均
    kl_per_sample = token_kl.sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
    
    return beta * kl_per_sample
```

### 小节 6：参考模型管理

```python
# 参考模型（π_ref）的创建和使用
class GRPOTrainer:
    def __init__(self, model, ...):
        # 深拷贝一份当前模型作为参考模型
        self.ref_model = create_reference_model(model)
        self.ref_model.eval()  # 设为评估模式
        
        # 冻结参考模型的所有参数
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def _get_ref_log_probs(self, input_ids, attention_mask):
        """获取参考模型的 log 概率（不计算梯度）"""
        with torch.no_grad():  # 不需要梯度，省显存
            ref_output = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_log_probs = self._get_log_probs(ref_output.logits, input_ids)
        return ref_log_probs
```

## 测验题

### Q1（代码阅读，4分）
在 `compute_policy_loss` 中，为什么 `old_log_probs` 要加 `.detach()`？如果不加会怎样？

**答案**：old_log_probs 是采样时记录的旧策略概率，是常数，不应该对它求梯度。不加 detach 会导致梯度流回旧策略的计算图，浪费计算资源且梯度不正确。（4分）

### Q2（对应理论，3分）
`compute_advantages` 中的 `+ 1e-8` 对应阶段二 Q6 的什么场景？

**答案**：对应全对/全错时 std=0 导致除零的场景。+1e-8 使结果为约 0 而非 NaN，这组数据不贡献梯度。（3分）

### Q3（代码 Bug，3分）
如果把 `ratio = torch.exp(log_ratio)` 改成 `ratio = new_log_probs / old_log_probs`，这是阶段二 Exam 3 哪道题的 bug？会导致什么问题？

**答案**：Exam 3 Q10 的 Bug 1。log 概率是负数，直接相除不等于概率的比值，而且 log 概率可以为 0 导致除零错误。正确做法是 log 空间相减再 exp。（3分）
