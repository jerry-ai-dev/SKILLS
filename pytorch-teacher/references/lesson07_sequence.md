# Lesson 7: 序列模型与词嵌入

## 教学目标
理解文本处理的基本流程，认识 RNN 的局限，为 Attention 做铺垫。

## 讲解要点

### 1. 文本 → 数字的流程
```
"I love pytorch" → 分词 → [I, love, pytorch] → 编号 → [1, 42, 567] → Embedding → 向量
```

### 2. 代码示例

```python
import torch
import torch.nn as nn

# ========== Embedding: 文字变向量 ==========
# 假设词汇表有 1000 个词，每个词用 64 维向量表示
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=64)

# 输入: 一句话，3个词的编号
word_ids = torch.tensor([42, 567, 1])  # "love", "pytorch", "I"
word_vectors = embedding(word_ids)
print(f"词编号: {word_ids.shape} → 词向量: {word_vectors.shape}")
# [3] → [3, 64]  每个词变成了 64维向量

# ========== 简单 RNN ==========
rnn = nn.RNN(input_size=64, hidden_size=128, batch_first=True)
# 输入: [batch=1, seq_len=3, embed_dim=64]
x = word_vectors.unsqueeze(0)  # 加上 batch 维度
output, hidden = rnn(x)
print(f"\nRNN 输出: {output.shape}")   # [1, 3, 128] 每个时间步的输出
print(f"RNN 隐藏: {hidden.shape}")     # [1, 1, 128] 最后的隐藏状态

# ========== LSTM (改进版 RNN) ==========
lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
output, (hidden, cell) = lstm(x)
print(f"\nLSTM 输出: {output.shape}")
print(f"LSTM 隐藏: {hidden.shape}")

# ========== 文本分类模型 ==========
class TextClassifier(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len] 词编号
        embedded = self.embedding(x)         # [batch, seq_len, embed_dim]
        output, (hidden, _) = self.lstm(embedded)  # hidden: [1, batch, hidden_dim]
        hidden = hidden.squeeze(0)           # [batch, hidden_dim]
        logits = self.fc(hidden)             # [batch, num_classes]
        return logits

model = TextClassifier()
# 模拟输入: 2句话，每句10个词
fake_input = torch.randint(0, 5000, (2, 10))
output = model(fake_input)
print(f"\n文本分类: 输入{fake_input.shape} → 输出{output.shape}")

# ========== RNN 的问题（为 Attention 铺垫）==========
print("\n" + "="*50)
print("RNN 的核心问题:")
print("1. 顺序处理 → 不能并行 → 训练慢")
print("2. 长距离依赖 → 信息在传递中会丢失")
print("3. 例如: 'The cat, which sat on the mat, was hungry'")
print("   → RNN 处理到 'was' 时，可能已经忘了主语是 'cat'")
print("4. 解决方案 → Attention 机制！(下节课)")
```

## 测验题目

### Q1
Embedding(1000, 64) 实际上维护了什么？

**答案**: 一个 1000×64 的参数矩阵。每一行对应一个词的 64 维向量表示。这些向量是可学习的参数。

### Q2
RNN 为什么不适合处理长序列？

**答案**: 因为信息必须沿时间步逐步传递，长序列中前面的信息会逐渐"稀释"和遗忘（梯度消失问题）。LSTM 缓解了但没有根本解决。

### Q3 (思考题)
如果要翻译 "I love you" → "我爱你"，RNN 处理到"你"时需要回顾原句的哪个词？这说明了什么？

**答案**: 需要回顾 "you"。这说明解码器在每一步需要"关注"输入序列的不同位置——这正是 Attention 的核心思想。

### Q4 (选择题)
LSTM 相比普通 RNN 多了什么关键机制？
A) 更多的隐藏层  B) 门控机制（遗忘门、输入门、输出门）  C) Attention  D) 卷积操作

**答案**: B — LSTM 通过三个门（遗忘门决定丢弃什么、输入门决定写入什么、输出门决定输出什么）来控制信息的流动，从而缓解梯度消失问题，更好地捕捉长距离依赖。

### Q5 (代码题)
已知词汇表大小 vocab_size=1000，嵌入维度 embed_dim=64。写出用 `nn.Embedding` 将 token ID 列表 `[0, 3, 7]` 转换为嵌入向量的代码，并说出输出 shape。

**参考答案**:
```python
embed = nn.Embedding(1000, 64)
ids = torch.tensor([0, 3, 7])
vectors = embed(ids)
print(vectors.shape)  # torch.Size([3, 64])  — 3个token，每个64维向量
```

## 实践任务
1. 用简单的正面/负面词列表构造一个小型情感分析数据集
2. 训练上面的 TextClassifier
3. 观察 Embedding 层学到了什么（比较"good"和"great"的向量距离）
