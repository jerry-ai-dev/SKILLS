# 期末考试：PyTorch 综合能力测评 (覆盖 Lesson 1-12)

## 考试说明
- 共 15 道题，满分 150 分（换算为百分制：得分 ÷ 1.5）
- 覆盖全部 12 课内容，重点考察核心概念的融会贯通
- 题型：选择题 × 5、判断题 × 2、代码题 × 3、简答题 × 3、综合题 × 2
- 评分标准：90+ 优秀 🌟 | 70-89 良好 👍 | 60-69 及格 ✅ | <60 需复习 📖

## 考题

### Q1 (选择题 · 10分) [Tensor]
以下哪种方式可以把张量从 GPU 移到 CPU？
A) `tensor.cpu()`
B) `tensor.to('cpu')`
C) `tensor.detach()`
D) A 和 B 都可以

**答案**: D
**解析**: `.cpu()` 和 `.to('cpu')` 都能将张量移到 CPU。`.detach()` 只是从计算图中分离，不改变设备。

### Q2 (选择题 · 10分) [Autograd]
以下关于 `loss.backward()` 的说法，哪个是错误的？
A) 会计算 loss 对所有 requires_grad=True 的叶子节点的梯度
B) 梯度存储在各叶子节点的 .grad 属性中
C) 默认情况下，计算图会在 backward() 后被释放
D) backward() 会自动更新模型参数

**答案**: D
**解析**: backward() 只计算梯度，不更新参数。更新参数由 optimizer.step() 完成。这是一个经典的初学者误区。

### Q3 (选择题 · 10分) [CNN]
输入 [1, 3, 64, 64] → Conv2d(3,32,3,p=1) → ReLU → MaxPool(2) → Conv2d(32,64,3,p=1) → ReLU → MaxPool(2)，最终输出 shape 是？
A) [1, 64, 64, 64]
B) [1, 64, 32, 32]
C) [1, 64, 16, 16]
D) [1, 32, 16, 16]

**答案**: C
**解析**: 64×64 → Conv(p=1) → 64×64 → Pool(2) → 32×32 → Conv(p=1) → 32×32 → Pool(2) → 16×16。通道 3→32→64。

### Q4 (选择题 · 10分) [Transformer]
以下关于 Transformer 的说法，哪个是错误的？
A) Self-Attention 可以捕捉任意距离的依赖关系
B) 位置编码是 Transformer 处理顺序信息的方式
C) Transformer 的计算复杂度随序列长度线性增长
D) Transformer 可以完全并行计算

**答案**: C
**解析**: Transformer 的 Self-Attention 复杂度是 $O(N^2)$（N 为序列长度），因为每个 token 都要和所有其他 token 计算注意力。这也是长文本处理的瓶颈。

### Q5 (选择题 · 10分) [Pretrained]
以下哪种模型是 Encoder-only 架构？
A) GPT-2
B) BERT
C) T5
D) GPT-4

**答案**: B
**解析**: BERT = Encoder-only（双向），GPT 系列 = Decoder-only（单向），T5 = Encoder-Decoder（完整结构）。

### Q6 (判断题 · 10分) [Training]
训练集 loss 持续下降，但验证集 loss 先降后升，说明模型发生了过拟合。(T/F)

**答案**: T
**解析**: 这是过拟合的经典表现。训练集上持续优化，但模型开始"记忆"训练数据的噪声，对新数据（验证集）的泛化能力下降。

### Q7 (判断题 · 10分) [Fine-tuning]
LoRA 微调时需要更新模型的所有参数。(T/F)

**答案**: F
**解析**: LoRA (Low-Rank Adaptation) 的核心思想就是冻结原始模型参数，只训练注入的低秩分解矩阵 A 和 B。通常只更新不到 1% 的参数。

### Q8 (代码题 · 10分) [Tensor + Autograd]
写出以下代码的完整输出，并解释每一步：
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * 2 + 1).sum()
y.backward()
print(y.item())
print(x.grad)
```

**参考答案**:
```
13.0
tensor([2., 2., 2.])
```
- `x * 2 + 1` = [3.0, 5.0, 7.0]
- `.sum()` = 15.0... 不对，2*1+1=3, 2*2+1=5, 2*3+1=7, sum=15.0

修正：y = sum(2x + 1) = 2×1+1 + 2×2+1 + 2×3+1 = 3 + 5 + 7 = 15.0
dy/dx_i = 2 对所有 i

输出：
```
15.0
tensor([2., 2., 2.])
```

**评分标准**: y 值正确 4分，梯度正确 4分，能解释推导过程 2分。

### Q9 (代码题 · 10分) [Dataset + Training]
完整写出一个训练流程的核心代码片段：创建 DataLoader → 遍历一个 epoch → 打印平均 loss。假设 model, criterion, optimizer, dataset 已定义。

**参考答案**:
```python
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
model.train()
total_loss = 0

for data, target in loader:
    output = model(data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

avg_loss = total_loss / len(loader)
print(f"Average Loss: {avg_loss:.4f}")
```
**评分标准**: DataLoader 创建正确 2分，训练五步骤顺序正确 5分，统计并打印 loss 正确 3分。

### Q10 (代码题 · 10分) [Attention]
给定 Q, K, V shape 均为 [1, 4, 8]（batch=1, 序列长度=4, 维度=8），写出带 causal mask 的 attention 计算过程。

**参考答案**:
```python
import torch
import torch.nn.functional as F

Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)

d_k = Q.size(-1)
scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [1, 4, 4]

# Causal mask
mask = torch.tril(torch.ones(4, 4))  # 下三角
scores = scores.masked_fill(mask == 0, float('-inf'))

weights = F.softmax(scores, dim=-1)  # [1, 4, 4]
output = torch.matmul(weights, V)     # [1, 4, 8]
```
**评分标准**: scores 计算正确 3分，mask 生成和应用正确 4分，最终输出正确 3分。

### Q11 (简答题 · 10分) [nn.Module 综合]
对比 `model.train()` 和 `model.eval()` 的区别。它们各自影响哪些层的行为？为什么推理时必须用 `model.eval()`？

**参考答案**:
- `model.train()`: 开启训练模式。Dropout 随机丢弃神经元，BatchNorm 用当前 batch 的均值/方差。
- `model.eval()`: 开启评估模式。Dropout 关闭（使用所有神经元），BatchNorm 使用训练期间积累的全局均值/方差。
- 推理时必须用 eval()，否则 Dropout 会随机丢弃导致结果不一致，BatchNorm 用单个 batch 统计量不准确。

**评分标准**: Dropout 行为区别 3分，BatchNorm 行为区别 3分，推理原因 4分。

### Q12 (简答题 · 10分) [Tokenizer + Pretrained]
解释 BPE (Byte Pair Encoding) tokenizer 的工作原理。为什么现代 LLM 使用子词(subword)分词而不是按词或按字符分词？

**参考答案**:
BPE 流程：
1. 初始化：将所有文本拆成字符级 token
2. 统计所有相邻 token 对的出现频率
3. 合并频率最高的一对为新 token
4. 重复步骤 2-3 直到词汇表达到目标大小

为什么用子词而不是词级/字符级：
- **词级问题**: 词汇表巨大，无法处理未登录词（OOV）。"unhappiness" 如果没见过就无法处理
- **字符级问题**: 序列太长（一个句子变成几十上百个 token），效率低，且难以捕捉语义
- **子词优势**: 平衡了两者。常见词保持完整（"the"），罕见词拆成有意义的子词（"un" + "happi" + "ness"），词汇表大小可控

**评分标准**: BPE 流程 4分，解释子词优势 6分。

### Q13 (简答题 · 10分) [Sequence → Attention]
请用一个生动的类比来解释 Attention 机制中 Query、Key、Value 的角色。

**参考答案**:
类比"图书馆找书"：
- **Query (查询)**: 你心中的问题——"我想找关于猫咪养护的书"
- **Key (键)**: 每本书封面上的标签/关键词——"宠物护理"、"烹饪"、"编程"
- **Value (值)**: 书的实际内容

工作流程：
1. 拿你的 Query 和所有书的 Key 做匹配（点积→相似度）
2. 越相关的 Key 得分越高（softmax→权重）
3. 用这些权重去取各本书的 Value 内容进行加权混合

**评分标准**: 类比合理且易懂 5分，清晰说明三者关系 5分。接受其他合理类比。

### Q14 (综合题 · 10分) [GPT 全链路]
描述从用户输入 "Hello" 到 GPT 输出一段文字的完整数据流经过程。请按顺序列出每个阶段及其输入输出。

**参考答案**:
1. **Tokenizer 编码**: "Hello" → [15496]（token ID）
2. **Token Embedding**: [15496] → [1, 1, 768]（嵌入向量，假设 d_model=768）
3. **加位置编码**: 嵌入 + PE → [1, 1, 768]
4. **通过 N 个 Transformer Block**:
   - Masked Self-Attention → Add & Norm
   - FFN → Add & Norm
   - 输出 [1, 1, 768]
5. **LM Head (线性层)**: [1, 1, 768] → [1, 1, vocab_size]（每个词的 logits）
6. **采样/Greedy**: 从 logits 中选一个 token，如 "," → token ID 11
7. **拼接并重复**: 输入变为 [15496, 11]，重复步骤 2-6
8. **Tokenizer 解码**: 所有 token ID → 最终文本 "Hello, how are you?"

**评分标准**: 正确描述主要阶段 6分，shape 变化正确 2分，提到自回归循环 2分。

### Q15 (综合题 · 10分) [Fine-tuning 综合]
你有一个预训练的 GPT-2 模型，想让它专门写中文古诗。请设计完整的微调方案，包括：数据准备、微调策略选择、训练细节、评估方法。

**参考答案**:
1. **数据准备**:
   - 收集大量中文古诗（《全唐诗》《全宋词》等），每首一个样本
   - 用 GPT-2 的 tokenizer（或换中文 tokenizer）编码
   - 格式化为 "诗题：xxx\n诗文：xxx" 的统一模板

2. **微调策略**:
   - 推荐使用 **LoRA/QLoRA**（显存有限时）或全参数微调（资源充足时）
   - LoRA 建议 rank=8-16，target_modules=["q_proj", "v_proj"]

3. **训练细节**:
   - 学习率：2e-5 ~ 5e-5（比预训练小 10-100 倍）
   - Batch size：根据显存调整（如 8-16）
   - Epochs：3-10 轮，配合 early stopping
   - 损失函数：Cross-Entropy（Next Token Prediction）

4. **评估方法**:
   - Perplexity（困惑度）：越低越好
   - 人工评估：韵律、意境、格式是否符合古诗要求
   - 生成样本展示：给不同开头生成，检查质量

**评分标准**: 数据准备 3分，微调策略 3分，训练细节合理 2分，评估方法 2分。

## 薄弱点诊断（综合）

| 知识模块 | 对应题目 | 掌握程度 |
|---------|---------|---------|
| **基础**: Tensor 基本操作 | Q1, Q8 | |
| **基础**: Autograd 梯度计算 | Q2, Q8 | |
| **基础**: nn.Module 使用 | Q11 | |
| **基础**: 训练循环 | Q6, Q9 | |
| **实战**: DataLoader 使用 | Q9 | |
| **实战**: CNN shape 推导 | Q3 | |
| **核心**: Attention 机制 | Q10, Q13 | |
| **核心**: Transformer 架构 | Q4 | |
| **核心**: GPT 生成流程 | Q14 | |
| **前沿**: 预训练与 Tokenizer | Q5, Q12 | |
| **前沿**: 微调技术 | Q7, Q15 | |

## 成绩单模板

```
╔══════════════════════════════════════════════════════╗
║           🎓 PyTorch 期末考试成绩单                    ║
╠══════════════════════════════════════════════════════╣
║  考生: ________                                      ║
║  日期: ________                                      ║
║                                                      ║
║  各题得分:                                            ║
║  Q1[  ] Q2[  ] Q3[  ] Q4[  ] Q5[  ]                ║
║  Q6[  ] Q7[  ] Q8[  ] Q9[  ] Q10[  ]               ║
║  Q11[  ] Q12[  ] Q13[  ] Q14[  ] Q15[  ]           ║
║                                                      ║
║  总分: ____/150  百分制: ____                         ║
║  等级: ________________                              ║
║                                                      ║
║  强项: ________________                              ║
║  薄弱: ________________                              ║
║  建议: ________________                              ║
╚══════════════════════════════════════════════════════╝
```
