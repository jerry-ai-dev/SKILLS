# Exam 3: 期末综合考试

## 考试说明
- **范围**：Lesson 1-11 全部内容（含 Lesson 11 DPOTrainer）
- **题数**：15 题，满分 100 分
- **分布**：
  - 代码阅读题 5 道（每题 7 分 = 35 分）
  - 架构理解题 4 道（每题 7 分 = 28 分）
  - 实战设计题 3 道（每题 7 分 = 21 分）
  - 对比分析题 2 道（每题 8 分 = 16 分）

---

## 代码阅读题（5 × 7分 = 35分）

### Q1【代码阅读，7分】
以下代码实现了 GRPO 的优势计算。补全 `???` 部分：

```python
def compute_advantages(rewards):
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.???（dim=-1, keepdim=True)
    return (rewards - mean) / (std + ???)
```

**答案**：`std`, `1e-8`。（每个 3.5分）

### Q2【代码阅读，7分】
以下 SFT 代码有什么问题？

```python
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="./out", learning_rate=2e-5),
    train_dataset=dataset,
    # 没有传 tokenizer
)
```

**答案**：没有传入 tokenizer，SFTTrainer 无法对数据做 tokenize 和 chat template 处理，训练会报错或产生错误结果。应该加 `tokenizer=tokenizer`。（7分）

### Q3【代码阅读，7分】
以下 reward 函数在什么情况下会返回错误结果？

```python
def reward(completions, answer=None, **kwargs):
    return [1.0 if extract_last_number(c) == float(a) 
            else 0.0 for c, a in zip(completions, answer)]
```

**答案**：浮点数直接用 `==` 比较会有精度问题。例如答案是 0.1+0.2=0.3，但浮点计算得到 0.30000000000000004，`==` 会判为错。应该用 `math.isclose`。（7分）

### Q4【代码阅读，7分】
解释这段 KL 惩罚代码中 `.clamp(min=1)` 的作用：

```python
kl = token_kl.sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
```

**答案**：防止全 padding 序列（attention_mask 全为 0）时除以 0。clamp(min=1) 保证分母最小为 1。（7分）

### Q5【代码阅读，7分】
以下 GRPOTrainer 使用代码中，reward_funcs 为什么传的是列表而不是单个函数？

```python
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[accuracy_reward, format_reward],
    ...
)
```

**答案**：TRL 支持多个奖励函数组合使用，各自独立打分后加权求和。这样可以同时奖励"答案正确"和"格式规范"两个维度，比单一奖励更灵活。（7分）

---

## 架构理解题（4 × 7分 = 28分）

### Q6【架构理解，7分】
画出从"原始预训练模型"到"可部署的数学推理模型"的完整 pipeline，标注每步用到的工具/库。

**答案**：
预训练模型 → [TRL SFTTrainer] → SFT 模型 → [TRL GRPOTrainer + reward_funcs] → RL 模型 → [evaluate.py] → 准确率报告 → 部署。（7分）

### Q7【架构理解，7分】
TRL、Open-R1、SimpleRL-Zoo 三者的定位有什么区别？各适合什么用户？

**答案**：
- TRL：通用工具库，适合工程师和研究者
- Open-R1：完整 pipeline，适合复现 R1 论文
- SimpleRL-Zoo：轻量实验，适合资源有限的研究者
（每个 2.3分）

### Q8【架构理解，7分】
GRPO 训练中需要同时加载策略模型和参考模型。有哪些方法可以减少这带来的显存压力？

**答案**：(1) 参考模型用 8-bit 量化加载；(2) 使用 LoRA 只训练少量参数，参考模型共享基础参数；(3) 参考模型放到 CPU（速度慢但省显存）；(4) gradient_checkpointing 减少激活值。（7分）

### Q9【架构理解，7分】
为什么 GRPO 的学习率（~1e-6）比 SFT（~2e-5）小很多？

**答案**：RL 训练有反馈循环——策略更新影响下一步采样的分布，学习率太大会导致策略剧烈变化，采样分布突变，训练崩溃。SFT 是稳定的监督学习，数据分布不受模型影响。（7分）

---

## 实战设计题（3 × 7分 = 21分）

### Q10【实战设计，7分】
你要训练一个代码生成模型（不是数学推理），需要修改哪些组件？列出至少 3 个需要改的地方。

**答案**：(1) SFT 数据集换成代码数据（如 CodeAlpaca）；(2) reward 函数改为执行代码检查输出正确性；(3) 评估 benchmark 换成 HumanEval/MBPP；(4) extract_answer 改为提取代码块。（7分）

### Q11【实战设计，7分】
你的 GRPO 训练中 reward/mean 持续为 0（模型总是答错），列出 3 个排查步骤。

**答案**：(1) 检查 reward 函数本身是否有 bug（手动测几个样本）；(2) 检查模型生成的回答是否有内容（可能生成空字符串或乱码）；(3) 检查 SFT 基线模型的准确率是否太低（如果基线是 0%，GRPO 没法学）。（7分）

### Q12【实战设计，7分】
设计一个实验来验证"SFT 冷启动对 GRPO 的必要性"。需要几组实验？分别是什么？

**答案**：至少 3 组：(1) SFT → GRPO（标准流程）；(2) 直接 GRPO（无 SFT，R1-Zero 风格）；(3) 只 SFT（无 GRPO）。比较三组在测试集上的准确率和输出质量。（7分）

---

## 对比分析题（2 × 8分 = 16分）

### Q13【对比分析，8分】
对比 SFTTrainer、GRPOTrainer 与 DPOTrainer 在 5 个维度上的核心差异（训练目标、数据需求、模型数量、loss 计算、关键超参）。

**答案**：

| 维度 | SFTTrainer | GRPOTrainer | DPOTrainer |
|------|-----------|-------------|------------|
| 训练目标 | 模仿标注数据 | 最大化奖励 | 提高 chosen 相对于 rejected 的偏好概率 |
| 数据需求 | prompt + answer | 只需 prompt | prompt + chosen + rejected |
| 模型数量 | 1 个 | 2 个（策略+参考） | 2 个（策略+参考，reference_free 时 1 个） |
| Loss | cross_entropy | PPO-Clip + KL | `-logsigmoid(β * (Δ_policy - Δ_ref))` |
| 关键超参 | lr, epochs | G, β, ε, temperature | β, loss_type, max_length |

（每行 1.6分）

### Q14【对比分析，8分】
完成阶段三学习后，你认为从理论到工程实现，最大的 gap 是什么？结合你在阶段二考试中代码题的表现回答。

**答案**：开放题。好的回答应该提到：(1) 理论公式到代码的转化（如 log 空间的 exp 技巧）；(2) 工程细节（防除零、detach、mask 处理）；(3) 调参经验（β、G、lr 的配合）。（8分）

---

## 期末成绩单模板

```
╔══════════════════════════════════════════════╗
║       开源项目研读 — 期末成绩单               ║
╠══════════════════════════════════════════════╣
║  代码阅读:  5题        __/35分               ║
║  架构理解:  4题        __/28分               ║
║  实战设计:  3题        __/21分               ║
║  对比分析:  2题        __/16分               ║
╠══════════════════════════════════════════════╣
║  总分:  __/100  等级: ______                 ║
╠══════════════════════════════════════════════╣
║  完成时间: 2026-XX-XX                        ║
║  下一站:   阶段四 — 实战与论文撰写            ║
║  目标项目: SFT + GRPO 数学推理 (Qwen-1.5B)   ║
╚══════════════════════════════════════════════╝
```
