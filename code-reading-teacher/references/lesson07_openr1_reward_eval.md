# Lesson 7: Open-R1 奖励函数 & 评估体系

## 学习目标
- 读懂 Open-R1 的 rewards.py 完整实现
- 理解端到端评估流程
- 掌握从 SFT → GRPO → 评估的完整 pipeline

## 知识小节

### 小节 1：Open-R1 的 rewards.py

```python
# src/open_r1/rewards.py（核心逻辑简化版）

import re
import math

def accuracy_reward(completions, answer, **kwargs):
    """准确率奖励：答案正确 +1，错误 0"""
    rewards = []
    for completion, gt in zip(completions, answer):
        # 从模型输出中提取答案
        predicted = extract_answer(completion)
        # 与标准答案比较
        if predicted is not None and math.isclose(float(predicted), float(gt), rel_tol=1e-5):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    """格式奖励：检查是否按要求使用 <think> 和 <answer> 标签"""
    rewards = []
    for completion in completions:
        score = 0.0
        # 检查推理标签
        if "<think>" in completion and "</think>" in completion:
            score += 0.5
        # 检查答案标签
        if "<answer>" in completion and "</answer>" in completion:
            score += 0.5
        rewards.append(score)
    return rewards

def extract_answer(text):
    """从文本中提取答案"""
    # 策略 1：提取 <answer>...</answer> 中的内容
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        if numbers:
            return numbers[-1]
    
    # 策略 2：提取 \\boxed{...} 中的内容（LaTeX 格式）
    match = re.search(r'\\boxed\{(.*?)\}', text)
    if match:
        numbers = re.findall(r'-?\d+\.?\d*', match.group(1))
        if numbers:
            return numbers[-1]
    
    # 策略 3：提取最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None

# 奖励函数注册表
REWARD_FUNCS = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

def get_reward_funcs(names):
    """根据名称列表返回奖励函数列表"""
    return [REWARD_FUNCS[name] for name in names]
```

### 小节 2：答案提取的工程细节

答案提取是奖励函数最容易出 bug 的地方：

```python
# 常见坑和处理方法

# 坑 1: "360" vs "360.0" vs "360.00"
# 解决: 用 math.isclose 而不是 ==
math.isclose(360.0, 360, rel_tol=1e-5)  # True

# 坑 2: 模型输出 "答案是 -3" 但正则只匹配正数
# 解决: 正则包含负号 r'-?\d+\.?\d*'

# 坑 3: 模型输出多个数字 "3+5=8, 所以答案是8"
# 解决: 优先取 <answer> 标签内的，其次取最后一个数字

# 坑 4: 模型输出 LaTeX "答案是 $\frac{1}{2}$"
# 解决: 用 sympy 解析或限制输出为小数
```

### 小节 3：评估框架

```python
# src/open_r1/evaluate.py（简化版）

def evaluate_model(model, tokenizer, dataset, num_samples=500):
    """评估模型在测试集上的准确率"""
    correct = 0
    total = 0
    
    for example in dataset[:num_samples]:
        prompt = example["prompt"]
        ground_truth = example["answer"]
        
        # 用 greedy decoding 生成（评估时不需要多样性）
        output = model.generate(
            tokenizer(prompt, return_tensors="pt").input_ids,
            max_new_tokens=512,
            do_sample=False,  # greedy！评估时用贪心
        )
        
        completion = tokenizer.decode(output[0], skip_special_tokens=True)
        predicted = extract_answer(completion)
        
        if predicted and math.isclose(float(predicted), float(ground_truth), rel_tol=1e-5):
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy
```

**注意**：评估时用 `do_sample=False`（贪心），训练时用 `do_sample=True`（随机采样）。

### 小节 4：端到端 Pipeline

```bash
# 完整的 Open-R1 训练流程

# Step 1: SFT（在推理数据上微调）
accelerate launch src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config.yaml

# Step 2: GRPO（用 SFT 模型作为起点做 RL）
accelerate launch src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config.yaml \
    --model_name_or_path ./output/sft  # 用 SFT 的输出作为 GRPO 的输入

# Step 3: 评估
python src/open_r1/evaluate.py \
    --model_path ./output/grpo \
    --dataset gsm8k \
    --split test
```

### 小节 5：评估指标和 Benchmark

| Benchmark | 内容 | 难度 |
|-----------|------|------|
| GSM8K | 小学数学应用题 | ⭐⭐ |
| MATH | 高中数学竞赛 | ⭐⭐⭐⭐ |
| AIME | 美国数学邀请赛 | ⭐⭐⭐⭐⭐ |

典型的实验结果对比表：

| 模型 | GSM8K | MATH |
|------|-------|------|
| Qwen2.5-1.5B（原始） | 45% | 20% |
| + SFT | 65% | 35% |
| + SFT + GRPO | 72% | 42% |

## 测验题

### Q1（代码阅读，4分）
`extract_answer` 函数的三种提取策略的优先级是什么？为什么要有多种策略？

**答案**：优先级：(1) `<answer>` 标签 > (2) `\boxed{}` LaTeX > (3) 最后一个数字。需要多种策略是因为不同模型的输出格式不同，训练初期模型可能还没学会用标签，需要兜底方案。（4分）

### Q2（工程实践，3分）
评估时为什么用 `do_sample=False` 而训练时用 `do_sample=True`？

**答案**：评估时要测模型的"最佳表现"，贪心生成最可能的答案，结果可复现。训练时需要多样性（G条不同回答），必须随机采样才能产生差异，否则优势全为0。（3分）

### Q3（Pipeline 理解，3分）
如果跳过 SFT 直接做 GRPO（像 DeepSeek R1-Zero 那样），可能出现什么问题？

**答案**：模型输出格式混乱（多语言混杂、没有推理结构），虽然 RL 最终可能学到推理能力，但可读性差且训练不稳定。这是阶段二 Q15 和 DeepSeek R1 论文的核心发现。（3分）
