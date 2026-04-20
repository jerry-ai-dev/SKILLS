# Lesson 12: 微调与 AI 前沿展望

## 教学目标
理解微调的概念和方法，了解 AI 前沿方向。

## 讲解要点

### 1. 为什么要微调
- 预训练模型: 通用能力强，但特定任务不够精确
- 微调: 在你的数据上继续训练，让模型适应特定任务
- 类比: 大学毕业生（预训练）→ 岗位培训（微调）→ 特定工作

### 2. 代码示例

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# ========== 简单微调示例 ==========
# 用 DistilBERT 微调一个情感分类器

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 模拟数据
texts = [
    "This product is amazing!", "Terrible experience", 
    "I love it so much", "Worst purchase ever",
    "Highly recommended", "Complete waste of money",
    "Fantastic quality", "Very disappointing",
] * 10  # 重复增加数据量

labels = [1, 0, 1, 0, 1, 0, 1, 0] * 10

# 数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.encodings = tokenizer(texts, truncation=True, padding=True, 
                                    max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

dataset = TextDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 微调（冻结部分层以加速）
# 冻结 embedding 和前 4 层
for name, param in model.named_parameters():
    if 'classifier' not in name and 'layer.5' not in name:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total:.1%})")

# 训练
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

# 测试
model.eval()
test_texts = ["This is wonderful!", "I hate this thing"]
inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    preds = outputs.logits.argmax(-1)
for text, pred in zip(test_texts, preds):
    print(f"'{text}' → {'Positive' if pred==1 else 'Negative'}")
```

### 3. LoRA 简介（高效微调）

```python
# LoRA 的核心想法 (伪代码)
# 原始: output = x @ W (W 是大矩阵, 如 768×768)
# LoRA: output = x @ W + x @ A @ B
#   A: 768×4 (低秩), B: 4×768 (低秩)
#   只训练 A 和 B，W 冻结！
#   参数量: 768×4 + 4×768 = 6144 << 768×768 = 589824

class LoRALayer(nn.Module):
    """LoRA: Low-Rank Adaptation"""
    def __init__(self, original_layer, rank=4):
        super().__init__()
        self.original = original_layer
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # 冻结原始权重
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # 低秩分解
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
    
    def forward(self, x):
        original_out = self.original(x)
        lora_out = x @ self.lora_A @ self.lora_B
        return original_out + lora_out

# 只需训练 2×rank×dim 个参数！
print(f"\nLoRA 示例:")
print(f"  原始参数: {768*768:,}")
print(f"  LoRA 参数(rank=4): {768*4 + 4*768:,}")
print(f"  节省: {1 - (768*4*2)/(768*768):.1%}")
```

## AI 前沿方向概览

### RLHF (人类反馈强化学习)
```
预训练 GPT → SFT (监督微调) → RLHF (人类偏好对齐)
                                ↑
                         这就是 ChatGPT 的关键步骤
```
- 人类对模型输出排序 → 训练奖励模型 → 用 RL 优化
- 让模型的回答更有帮助、更安全、更诚实

### 多模态 (Vision + Language)
- 图片 + 文字 → 统一理解
- 如 GPT-4V、Claude 的看图能力
- 核心: 图像和文本都编码为向量，在 Transformer 中统一处理

### AI Agent
- LLM + 工具使用 + 规划能力
- 不只是"回答问题"，而是"完成任务"
- 如: 自动写代码、搜索信息、操作电脑

### 开源生态
- LLaMA (Meta), Qwen (阿里), DeepSeek 等
- 开源模型能力快速提升
- 社区微调、量化、部署

## 测验题目

### Q1
微调和从零训练的区别？

**答案**: 微调从预训练权重开始，只需少量数据和计算。从零训练从随机初始化开始，需要大量数据和计算。微调利用了预训练学到的通用知识。

### Q2
LoRA 为什么能大幅减少训练参数？

**答案**: LoRA 假设微调时权重变化是低秩的，用两个小矩阵 A×B 近似大矩阵的变化。只训练 A 和 B，冻结其他所有参数。

### Q3 (开放题)
你最想用 AI 做什么？可以用今天学到的知识实现吗？

### Q4 (选择题)
RLHF 中的四个字母分别代表什么？
A) Recurrent Learning from Human Feedback  B) Reinforcement Learning from Human Feedback  C) Regression Learning from Human Features  D) Recursive Learning from Hidden Features

**答案**: B — Reinforcement Learning from Human Feedback（从人类反馈中强化学习）。核心思路是先训练一个奖励模型来模拟人类偏好，再用强化学习（如 PPO）优化语言模型使其输出更符合人类期望。

### Q5 (思考题)
全参数微调 vs LoRA 微调，各自适合什么场景？如果你只有一张 8GB 显存的消费级显卡，应该选哪种？

**答案**: 全参数微调更新所有参数，效果可能更好但需要大量显存和数据（适合有充裕资源的场景）。LoRA 只训练极少量参数（通常不到原模型的 1%），显存占用小、训练快，适合资源有限的场景。8GB 显卡应选 LoRA（甚至 QLoRA = 量化 + LoRA），这是个人开发者微调大模型的首选方案。

## 毕业项目建议

选择一个感兴趣的方向：

1. **文本生成器**: 用 mini-GPT 训练一个特定风格的文本生成器（如古诗、代码）
2. **情感分析工具**: 微调 BERT 做特定领域的情感分析
3. **图像分类器**: 用 CNN 或预训练模型分类自己的图片
4. **聊天机器人**: 用 GPT-2 微调一个特定领域的对话模型
5. **自由创作**: 结合课程内容做任何你感兴趣的项目
