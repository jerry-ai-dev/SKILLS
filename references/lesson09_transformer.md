# Lesson 9: Transformer 架构

## 教学目标
理解 Transformer 的完整结构，搭建一个 mini-Transformer。

## 讲解要点

### 1. Transformer = Attention + 一些巧妙设计
```
Transformer Block:
    输入 → [Multi-Head Attention] → 残差连接 → LayerNorm
         → [Feed-Forward Network] → 残差连接 → LayerNorm → 输出
```

### 2. 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== Multi-Head Attention ==========
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 投影并拆分成多个头
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q,K,V: [batch, heads, seq_len, d_k]
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        # 合并多个头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)

# ========== Position Encoding ==========
class PositionalEncoding(nn.Module):
    """让模型知道每个词的位置"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ========== Transformer Block ==========
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-Attention + 残差 + LayerNorm
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-Forward + 残差 + LayerNorm
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

# ========== Mini Transformer ==========
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, 
                 num_layers=2, d_ff=256, num_classes=2, max_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len] 词编号
        x = self.embedding(x)      # [batch, seq_len, d_model]
        x = self.pos_encoding(x)   # 加上位置信息
        
        for block in self.blocks:
            x = block(x)
        
        # 取第一个位置的输出做分类（类似BERT的[CLS]）
        x = x[:, 0]                # [batch, d_model]
        return self.classifier(x)  # [batch, num_classes]

# 使用
model = MiniTransformer(vocab_size=5000, num_classes=2)
fake_input = torch.randint(0, 5000, (4, 20))  # 4句话，每句20词
output = model(fake_input)
print(f"Mini Transformer:")
print(f"  输入: {fake_input.shape}")
print(f"  输出: {output.shape}")
print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

# 结构一览
print(f"\n模型结构:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")
```

## Transformer 的关键设计

| 设计 | 作用 | 为什么重要 |
|------|------|-----------|
| Multi-Head | 多个角度关注不同模式 | 一个头看语法，一个头看语义 |
| 位置编码 | 告诉模型词的位置 | Attention 本身没有位置概念 |
| 残差连接 | 跳跃连接 | 让梯度直接流过，易于训练深层 |
| LayerNorm | 归一化 | 稳定训练，加速收敛 |
| FFN | 非线性变换 | 增加每层的表达能力 |

## 测验题目

### Q1
d_model=512, num_heads=8 时，每个头的维度 d_k 是多少？

**答案**: 512/8 = 64

### Q2
为什么 Transformer 需要位置编码？

**答案**: Self-Attention 是集合操作（不考虑顺序），"cat sat on mat" 和 "mat on sat cat" 会得到相同结果。位置编码显式添加位置信息。

### Q3 (思考题)
比较 Transformer 和 RNN 处理长度为 N 的序列：计算需要几步？

**答案**: RNN 需要 N 步（顺序），Transformer 只需 1 步（但计算量 O(N²)）。Transformer 能并行，所以在 GPU 上快得多。

### Q4 (选择题)
Transformer 中 Layer Normalization 的主要作用是？
A) 减少参数量  B) 稳定每层输入的分布，加速训练  C) 增加非线性  D) 实现注意力机制

**答案**: B — LayerNorm 将每一层的输入归一化到均值 0、方差 1，避免深层网络中各层输入分布剧烈变化（Internal Covariate Shift），从而加速收敛并稳定训练。

### Q5 (填空题)
正弦位置编码的公式中，偶数位使用 _____ 函数，奇数位使用 _____ 函数。这种设计的好处是可以让模型通过线性变换学习到 _____ 信息。

**答案**: sin，cos，相对位置。公式为 PE(pos, 2i) = sin(pos / 10000^(2i/d_model))，PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))。sin/cos 的组合使得任意两个位置的编码差可以用线性变换表示。

## 实践任务
1. 修改 MiniTransformer，将层数从 2 改为 4，观察参数量变化
2. 给 MultiHeadAttention 加入 causal mask
3. 打印每层 Attention 的权重分布
