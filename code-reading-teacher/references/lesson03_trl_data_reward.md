# Lesson 3: TRL 数据流水线 & Reward 设计

## 学习目标
- 理解 TRL 期望的数据格式和 Chat Template 机制
- 掌握 reward function 的接口规范
- 能为数学推理任务编写符合 TRL 接口的 reward 函数
- 理解训练日志中的关键指标

## 知识小节

### 小节 1：TRL 数据格式

TRL 支持多种数据格式，最常用的两种：

**格式 1：纯文本格式（dataset_text_field）**
```python
# 数据集中每条样本是一个完整的文本字符串
dataset = [
    {"text": "<|user|>\n小明有3个苹果\n<|assistant|>\n答案是3"},
    {"text": "<|user|>\n1+1等于几\n<|assistant|>\n答案是2"},
]
```

**格式 2：对话格式（messages）**
```python
# 数据集中每条样本是 messages 列表（推荐）
dataset = [
    {"messages": [
        {"role": "user", "content": "小明有3个苹果又买了5个，一共几个？"},
        {"role": "assistant", "content": "3+5=8，答案是8"},
    ]},
]
```

### 小节 2：Chat Template 机制

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "user", "content": "1+1=?"},
    {"role": "assistant", "content": "2"},
]

# apply_chat_template 把 messages 转成模型需要的格式
text = tokenizer.apply_chat_template(messages, tokenize=False)
print(text)
# 输出（Qwen 格式）:
# <|im_start|>user
# 1+1=?<|im_end|>
# <|im_start|>assistant
# 2<|im_end|>
```

**为什么需要 Chat Template？**
- 不同模型用不同的特殊 token 标记 user/assistant 的边界
- Qwen 用 `<|im_start|>/<|im_end|>`，Llama 用 `<|start_header_id|>/<|end_header_id|>`
- Chat Template 自动处理这些差异，写一份代码适配所有模型

### 小节 3：GRPO 的 Reward 函数接口

```python
# TRL GRPOTrainer 期望的 reward 函数签名

def my_reward_function(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
    """
    参数：
        completions: 模型生成的回答列表
        prompts: 对应的 prompt 列表（可选）
        **kwargs: 其他信息（如 ground_truth 答案）
    
    返回：
        rewards: 每条回答的奖励分数列表
    """
    rewards = []
    for completion in completions:
        score = evaluate(completion)
        rewards.append(score)
    return rewards
```

### 小节 4：数学推理的 Reward 函数实现

```python
import re

def accuracy_reward(completions, prompts=None, answer=None, **kwargs):
    """准确率奖励：答案正确 +1，错误 0"""
    rewards = []
    for completion, gt in zip(completions, answer):
        # 从模型输出中提取数字
        predicted = extract_answer(completion)
        if predicted is not None and abs(float(predicted) - float(gt)) < 1e-5:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    """格式奖励：有推理步骤 +0.5，有 <answer> 标签 +0.5"""
    rewards = []
    for completion in completions:
        score = 0.0
        # 检查是否有推理步骤（至少包含等号或"因为"）
        if "=" in completion or "因为" in completion or "所以" in completion:
            score += 0.5
        # 检查是否有答案标签
        if "<answer>" in completion and "</answer>" in completion:
            score += 0.5
        rewards.append(score)
    return rewards

def extract_answer(text):
    """从文本中提取最后一个数字作为答案"""
    # 优先提取 <answer>...</answer> 中的数字
    match = re.search(r'<answer>\s*(-?\d+\.?\d*)\s*</answer>', text)
    if match:
        return match.group(1)
    # 否则提取最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None
```

### 小节 5：GRPOTrainer 中使用 Reward 函数

```python
from trl import GRPOTrainer, GRPOConfig

# 可以传入多个 reward 函数，TRL 会把它们的分数加权求和
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        accuracy_reward,    # 准确率奖励
        format_reward,      # 格式奖励
    ],
    args=GRPOConfig(
        output_dir="./grpo_output",
        num_generations=8,             # G=8，每个 prompt 生成 8 条
        temperature=0.7,
        beta=0.1,                      # KL 惩罚系数 β
        max_completion_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        bf16=True,
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### 小节 6：训练日志关键指标

```
训练过程中要关注的指标：

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| reward/mean | 平均奖励 | 应该逐渐上升 |
| reward/std | 奖励标准差 | 不应该太小（否则全对/全错） |
| kl | KL 散度 | 一般 0.1~5，太大说明偏离太远 |
| loss | GRPO loss | 不一定要降（RL 的 loss ≠ 准确率） |
| completion_length | 生成长度 | 不应该持续增长（可能在灌水） |
```

## 测验题

### Q1（代码实现，4分）
写一个 reward 函数，检查模型输出是否包含 "因此" 或 "所以" 关键词。包含给 0.3 分，不包含给 0 分。要求符合 TRL 接口。

**答案**：
```python
def reasoning_keyword_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        if "因此" in c or "所以" in c:
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards
```
（接口正确 2分，逻辑正确 2分）

### Q2（概念理解，3分）
为什么 GRPOTrainer 的 generate 必须用 `do_sample=True` 而不能用 greedy？

**答案**：greedy 对同一个 prompt 只会生成一种回答，G 条全一样，奖励全相同，优势全为 0（阶段二 Q6 的全对/全错问题），无法提供梯度信号。必须随机采样才能产生多样性。（3分）

### Q3（指标分析，3分）
训练中 reward/mean 上升但 kl 也飙到 20+，说明什么？该怎么处理？

**答案**：说明模型在拿高分但偏离参考模型太远，可能出现 Reward Hacking 或能力退化（阶段二 Q7 的 β 太小场景）。应该增大 β 或降低学习率。（3分）
