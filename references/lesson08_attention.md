# Lesson 8: Attention 注意力机制

## 教学目标
从直觉到代码理解 Attention，能手写 Self-Attention。

## 讲解要点

### 1. Attention 的直觉
- 人看一张图片时不会均匀注视每个像素，而是"聚焦"关键区域
- 同理，处理"The cat sat on the mat"时，"sat"最应该关注"cat"
- Attention = 学会"该关注什么"

### 2. Query, Key, Value 类比
- 想象你在图书馆找书：
  - **Query**: 你的问题（"我想找关于猫的书"）
  - **Key**: 每本书的标签（"猫的习性"、"狗的训练"、"猫的食谱"）
  - **Value**: 书的内容
  - **过程**: Query 和每个 Key 比较相似度 → 越相似的书，你越仔细看 → 加权读取

### 3. 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== 从零实现 Scaled Dot-Product Attention ==========
def attention(Q, K, V, mask=None):
    """
    Q: [batch, seq_len, d_k]  查询
    K: [batch, seq_len, d_k]  键
    V: [batch, seq_len, d_v]  值
    """
    d_k = Q.size(-1)
    
    # 1. 计算相似度 (Q和K的点积)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: [batch, seq_len, seq_len]
    
    # 2. (可选) 应用mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 3. Softmax 得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

# 测试
batch_size, seq_len, d_model = 1, 4, 8
x = torch.randn(batch_size, seq_len, d_model)

# 简单情况：Q=K=V=x (Self-Attention)
output, weights = attention(x, x, x)
print(f"输入: {x.shape}")
print(f"输出: {output.shape}")
print(f"注意力权重:\n{weights.squeeze()}")
print(f"权重每行之和: {weights.squeeze().sum(dim=-1)}")  # 应该都是1

# ========== Self-Attention 层 ==========
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        output, attn_weights = attention(Q, K, V, mask)
        return output, attn_weights

# 使用
attn_layer = SelfAttention(d_model=64)
x = torch.randn(2, 10, 64)  # 2句话，每句10个词，64维
out, weights = attn_layer(x)
print(f"\nSelf-Attention: {x.shape} → {out.shape}")
print(f"注意力权重: {weights.shape}")  # [2, 10, 10] 每个词对其他词的关注度

# ========== 可视化注意力 ==========
print("\n===== 注意力权重可视化（文本形式）=====")
words = ["The", "cat", "sat", "on"]
# 假设一个简化的注意力矩阵
simple_attn = torch.tensor([
    [0.1, 0.6, 0.2, 0.1],  # "The" 主要关注 "cat"
    [0.1, 0.2, 0.5, 0.2],  # "cat" 主要关注 "sat"
    [0.2, 0.5, 0.1, 0.2],  # "sat" 主要关注 "cat"
    [0.1, 0.1, 0.3, 0.5],  # "on" 主要关注自己和"sat"
])
print(f"{'':>6}", end="")
for w in words:
    print(f"{w:>6}", end="")
print()
for i, w in enumerate(words):
    print(f"{w:>6}", end="")
    for j in range(len(words)):
        val = simple_attn[i][j].item()
        print(f"{val:>6.2f}", end="")
    print()
print("\n→ 'sat' 最关注 'cat'（谁做了sat的动作？）")
print("→ 这就是 Attention 学到的语义关系！")
```

## 核心公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$: 计算每对词之间的相似度
- $\sqrt{d_k}$: 缩放因子，防止点积过大
- softmax: 归一化为概率分布
- 乘以 V: 按注意力权重汇总信息

## 测验题目

### Q1
为什么要除以 $\sqrt{d_k}$？

**答案**: 当维度 d_k 很大时，点积值会很大，导致 softmax 输出接近 one-hot（梯度消失）。除以 √d_k 把值控制在合理范围。

### Q2 (代码题)
修改上面的 attention 函数，加入一个 causal mask（只能看到当前和之前的位置）。

**参考答案**:
```python
seq_len = 4
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
# [[1,0,0,0], [1,1,0,0], [1,1,1,0], [1,1,1,1]]
output, weights = attention(Q, K, V, mask=causal_mask)
```

### Q3
Self-Attention 和 Cross-Attention 的区别？

**答案**: Self-Attention 中 Q、K、V 来自同一个序列（自己和自己交互）。Cross-Attention 中 Q 来自一个序列，K、V 来自另一个序列（如翻译中解码器关注编码器的输出）。

### Q4 (选择题)
Attention 机制的输出是什么？
A) 一个标量分数  B) Q 和 K 的点积  C) Value 的加权求和  D) Softmax 概率分布

**答案**: C — Attention 输出 = softmax(QK^T / √d_k) × V，本质是用注意力权重对 Value 做加权求和，让模型"聚焦"于相关信息。

### Q5 (判断题)
Multi-Head Attention 中，不同的 head 可以学习到不同类型的关系模式（如一个关注语法，一个关注语义）。(T/F)

**答案**: T — 每个 head 有独立的 W_Q、W_K、W_V 投影矩阵，可以学到不同的注意力模式。实验中确实观察到不同 head 关注不同类型（如位置、语法、指代等）的关系。

## 实践任务
1. 对一段真实文本（如3-5词的句子）运行 Self-Attention，打印注意力权重矩阵
2. 分析权重：哪些词对之间的注意力最高？是否符合语义直觉？
3. 比较有/无缩放因子的效果
