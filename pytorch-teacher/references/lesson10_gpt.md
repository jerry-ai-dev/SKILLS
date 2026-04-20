# Lesson 10: 从 Transformer 到 GPT

## 教学目标
理解 GPT 的 Decoder-only 架构，搭建 mini-GPT 做文本生成。

## 讲解要点

### 1. GPT 的核心思路
- 训练目标: 给定前面的词，预测下一个词
- "The cat sat" → 预测 "on"
- Decoder-only: 只用 Transformer Decoder（带 causal mask）
- 与 Lesson 9 的分类不同，GPT 是"生成式"模型

### 2. 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== Mini GPT ==========
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, 
                 num_layers=4, d_ff=256, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享：embedding 和输出层共享参数
        self.head.weight = self.token_embedding.weight
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Token + Position Embedding
        positions = torch.arange(seq_len, device=x.device)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        
        # Causal Mask: 只能看到当前和之前的位置
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)  # [batch, seq_len, vocab_size]
        return logits
    
    def generate(self, start_tokens, max_new_tokens=50, temperature=1.0):
        """自回归生成"""
        self.eval()
        tokens = start_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(tokens)
                # 只取最后一个位置的预测
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token], dim=1)
        
        return tokens

# 需要上一课的 TransformerBlock（这里假设已定义）
# 为了独立运行，这里重新定义一个简化版：
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        if mask is not None:
            attn_mask = mask.bool().logical_not()  # PyTorch 用 True 表示"遮住"
            attn_out, _ = self.attn(x, x, x, attn_mask=~mask.bool() if mask is not None else None)
        else:
            attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# ========== 训练：字符级文本生成 ==========
# 准备数据
text = """To be or not to be that is the question.
Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune 
or to take arms against a sea of troubles and by opposing end them."""

# 字符级 tokenizer
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
vocab_size = len(chars)

# 编码文本
data = torch.tensor([char_to_idx[c] for c in text])
print(f"文本长度: {len(text)}, 词汇表大小: {vocab_size}")
print(f"示例编码: '{text[:10]}' → {data[:10].tolist()}")

# 创建训练数据
def create_sequences(data, seq_len=32):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+1:i+seq_len+1])
    return torch.stack(X), torch.stack(Y)

X, Y = create_sequences(data, seq_len=32)
print(f"训练样本数: {len(X)}")

# 创建模型
model = MiniGPT(vocab_size=vocab_size, d_model=64, num_heads=4, 
                num_layers=2, d_ff=128, max_len=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(100):
    logits = model(X)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# 生成文本
start = torch.tensor([[char_to_idx['T']]])
generated = model.generate(start, max_new_tokens=100, temperature=0.8)
generated_text = ''.join([idx_to_char[i.item()] for i in generated[0]])
print(f"\n生成的文本:\n{generated_text}")
```

## GPT 系列发展

| 模型 | 参数量 | 特点 |
|------|--------|------|
| GPT-1 | 117M | 证明了预训练+微调的路线 |
| GPT-2 | 1.5B | 证明了 scale up 的威力 |
| GPT-3 | 175B | Few-shot learning |
| GPT-4 | ~1.8T(估) | 多模态、强推理 |

**关键洞察**: GPT 的架构和我们写的 mini-GPT 原理完全一样，区别只在规模（参数量、数据量、计算量）。

## 测验题目

### Q1
Causal Mask 的作用是？不用会怎样？

**答案**: 确保在预测第 i 个位置时只能看到位置 0~i-1。不用的话模型会"偷看"未来的答案，训练时 loss 很低但生成时不会用。

### Q2
GPT 的训练目标是什么？用一句话描述。

**答案**: 给定前面所有的 token，最大化下一个 token 的概率（Next Token Prediction）。

### Q3
temperature 参数的作用？temperature → 0 和 → ∞ 分别会怎样？

**答案**: temperature 控制生成的随机性。→ 0 时变成贪心解码（总是选概率最高的词，确定性但可能重复）；→ ∞ 时趋近均匀分布（完全随机，不连贯）。

### Q4 (选择题)
GPT 使用的是 Transformer 的哪个部分？
A) 完整的 Encoder-Decoder  B) 仅 Encoder  C) 仅 Decoder  D) 既不是 Encoder 也不是 Decoder

**答案**: C — GPT 是 Decoder-only 架构，使用因果掩码的 Self-Attention，只能看到当前位置及之前的 token。BERT 则是 Encoder-only（双向）。

### Q5 (代码题)
实现 top-k 采样的核心逻辑：给定 logits 张量和 k 值，写出只保留概率最高的 k 个 token 并从中采样的代码。

**参考答案**:
```python
def top_k_sample(logits, k=5):
    # 只保留 top-k，其余设为 -inf
    values, indices = logits.topk(k)
    filtered = torch.full_like(logits, float('-inf'))
    filtered.scatter_(0, indices, values)
    # 转为概率并采样
    probs = torch.softmax(filtered, dim=0)
    return torch.multinomial(probs, 1)
```

## 实践任务
1. 用更长的文本（如一篇文章）训练 mini-GPT
2. 尝试不同 temperature 值，观察生成文本的变化
3. 改为 word-level（词级）而不是 char-level（字符级）
