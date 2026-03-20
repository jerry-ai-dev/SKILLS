# 阶段考试 3：Attention 与 Transformer (覆盖 Lesson 8-10)

## 考试说明
- 共 10 道题，满分 100 分
- 题型：选择题 × 3、判断题 × 2、代码题 × 2、简答题 × 2、推导题 × 1
- 评分标准：90+ 优秀 🌟 | 70-89 良好 👍 | 60-69 及格 ✅ | <60 需复习 📖

## 考题

### Q1 (选择题 · 10分) [Attention]
Scaled Dot-Product Attention 公式为 $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$，其中除以 $\sqrt{d_k}$ 的目的是？
A) 加速计算
B) 防止点积值过大导致 softmax 梯度消失
C) 降低参数量
D) 使 Q 和 K 的维度匹配

**答案**: B
**解析**: 当 $d_k$ 很大时，$QK^T$ 的值方差约为 $d_k$，值很大的区域 softmax 接近 one-hot，梯度几乎为 0。除以 $\sqrt{d_k}$ 将方差缩放到 1，使 softmax 输出更平滑，梯度更好。

### Q2 (选择题 · 10分) [Transformer]
d_model=256, num_heads=8，每个头的 Q、K、V 维度是？
A) 256
B) 128
C) 32
D) 8

**答案**: C
**解析**: d_k = d_model / num_heads = 256 / 8 = 32。每个头在更小的子空间上做注意力，最后拼接回 256 维。

### Q3 (选择题 · 10分) [GPT]
GPT 中 Causal Mask 是一个什么形状的矩阵？
A) 全 1 矩阵
B) 单位矩阵（对角线为 1）
C) 下三角矩阵（对角线及以下为 1）
D) 上三角矩阵（对角线及以上为 1）

**答案**: C
**解析**: 下三角矩阵确保位置 i 只能看到 0~i 位置（包括自己）。在被 mask 为 0 的位置上，attention score 被设为 -inf，softmax 后权重为 0。

### Q4 (判断题 · 10分) [Attention]
Self-Attention 中 Q、K、V 来自同一个输入，但它们经过了不同的线性变换（W_Q, W_K, W_V），所以实际上是不同的。(T/F)

**答案**: T
**解析**: 虽然 Q = XW_Q, K = XW_K, V = XW_V 都来自同一个 X，但三个投影矩阵是独立的可学习参数，使得 Q、K、V 扮演不同角色：Q 是"提问"，K 是"被查询的索引"，V 是"被检索的内容"。

### Q5 (判断题 · 10分) [Transformer]
Transformer 中的位置编码可以让模型分辨 "I love you" 和 "you love I" 的区别。(T/F)

**答案**: T
**解析**: Self-Attention 本身是集合操作，不区分顺序。加入位置编码后，每个位置的 token 嵌入中融合了位置信息，使得模型能够感知词序。没有位置编码的 Transformer 无法区分词序不同的句子。

### Q6 (代码题 · 10分) [Attention]
用 PyTorch 实现最基本的 Scaled Dot-Product Attention 函数（不含 mask）。输入 Q, K, V 的 shape 为 [batch, seq_len, d_k]。

**参考答案**:
```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [batch, seq_len, seq_len]
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)  # [batch, seq_len, d_k]
    return output, weights
```
**评分标准**: QK^T 计算正确 3分，缩放因子正确 2分，softmax 维度正确 2分，与 V 相乘正确 3分。

### Q7 (代码题 · 10分) [GPT]
写出生成 causal mask 的代码（给定序列长度 seq_len），并说明如何把它应用到 attention scores 上。

**参考答案**:
```python
def create_causal_mask(seq_len):
    # 下三角矩阵
    mask = torch.tril(torch.ones(seq_len, seq_len))  # [seq_len, seq_len]
    return mask

# 应用到 attention scores
seq_len = 5
mask = create_causal_mask(seq_len)
scores = torch.randn(seq_len, seq_len)  # 原始 attention scores
scores = scores.masked_fill(mask == 0, float('-inf'))  # mask 为 0 处填 -inf
weights = torch.softmax(scores, dim=-1)  # softmax 后 -inf 变成 0
```
**评分标准**: tril 生成正确 3分，masked_fill 使用正确 4分，解释 -inf→softmax→0 的原理 3分。

### Q8 (简答题 · 10分) [Transformer]
请解释 Transformer 中 Add & Norm（残差连接 + Layer Normalization）的作用。为什么深层 Transformer 离不开它？

**参考答案**:
- **残差连接 (Add)**: `output = sublayer(x) + x`。让梯度可以直接"跳过"子层传播，缓解深层网络的梯度消失问题。即使子层学到的东西很少，至少还有 x 保底，不会比没有这层更差。
- **Layer Normalization (Norm)**: 将每一层的输入归一化到均值 0、方差 1。稳定各层输入分布，加速收敛，避免数值不稳定。
- **为什么关键**: Transformer 通常有 6-96 层之深。没有残差连接，梯度会在反向传播中消失；没有 LayerNorm，各层输入分布剧烈波动，训练极不稳定。

**评分标准**: 残差连接作用 4分，LayerNorm 作用 3分，解释必要性 3分。

### Q9 (简答题 · 10分) [GPT]
解释 GPT 的"自回归生成"过程。给定 prompt "The cat"，GPT 是如何一步步生成后续文本的？

**参考答案**:
1. 输入 "The cat" → 模型预测下一个 token 的概率分布
2. 从概率分布中采样（或 greedy/top-k/top-p）得到一个 token，如 "sat"
3. 将 "The cat sat" 作为新输入 → 又预测下一个 token
4. 从分布中采样得到 "on"
5. 重复此过程直到生成 EOS token 或达到最大长度

核心特点：每次只生成一个 token，且只能看到之前的 token（因果性）。这就是为什么需要 Causal Mask。

**评分标准**: 描述逐步生成过程 5分，提到采样策略 2分，解释因果性 3分。

### Q10 (推导题 · 10分) [综合]
一个 Transformer Block 包含：Multi-Head Attention + FFN (两层 Linear)。
已知 d_model=512, num_heads=8, d_ff=2048，计算一个 Block 的参数量。

提示：Multi-Head Attention 有 4 个线性变换 (W_Q, W_K, W_V, W_O)；FFN 有 2 个线性变换 + 偏置；先忽略 LayerNorm 参数。

**参考答案**:
- **Multi-Head Attention**:
  - W_Q: 512 × 512 + 512 = 262,656
  - W_K: 512 × 512 + 512 = 262,656
  - W_V: 512 × 512 + 512 = 262,656
  - W_O: 512 × 512 + 512 = 262,656
  - 小计: 262,656 × 4 = **1,050,624**
- **FFN**:
  - Linear1: 512 × 2048 + 2048 = 1,050,624
  - Linear2: 2048 × 512 + 512 = 1,049,088
  - 小计: **2,099,712**
- **总计**: 1,050,624 + 2,099,712 = **3,150,336** ≈ 315万参数

**评分标准**: MHA 参数算对 4分，FFN 参数算对 4分，最终加总正确 2分。有过程给分。

## 薄弱点诊断

| 模块 | 对应题目 | 掌握程度 |
|------|---------|---------|
| Attention 原理与公式 | Q1, Q4, Q6 | |
| Multi-Head Attention | Q2, Q10 | |
| 位置编码 | Q5 | |
| Causal Mask | Q3, Q7 | |
| Transformer 结构 (Add&Norm, FFN) | Q8, Q10 | |
| GPT 自回归生成 | Q9 | |
