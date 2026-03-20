# 阶段考试 1：基础篇 (覆盖 Lesson 1-4)

## 考试说明
- 共 10 道题，满分 100 分
- 题型：选择题(10分/题) × 4、判断题(10分/题) × 2、代码题(10分/题) × 2、思考/简答题(10分/题) × 2
- 时间：不限时，但建议不要翻阅笔记（考验真实掌握程度）
- 评分标准：90+ 优秀 🌟 | 70-89 良好 👍 | 60-69 及格 ✅ | <60 需复习 📖

## 考题

### Q1 (选择题 · 10分) [Tensor]
以下哪个操作能将两个 shape 为 (3, 4) 的张量做矩阵乘法？
A) `a * b`
B) `a @ b`
C) `a @ b.T`
D) `torch.dot(a, b)`

**答案**: C
**解析**: 矩阵乘法要求第一个矩阵的列数 = 第二个矩阵的行数。(3,4) @ (4,3) = (3,3)。A 是逐元素乘法；B 维度不匹配会报错；D 只能用于一维向量。

### Q2 (选择题 · 10分) [Autograd]
以下代码的输出是什么？
```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x
y.backward()
print(x.grad)
```
A) tensor(3.0)  B) tensor(8.0)  C) tensor(11.0)  D) tensor(6.0)

**答案**: B
**解析**: y = x² + 2x，dy/dx = 2x + 2 = 2×3 + 2 = 8.0

### Q3 (选择题 · 10分) [nn.Module]
一个 `nn.Linear(256, 64)` 层，加上偏置(bias)，总共有多少个可训练参数？
A) 256 × 64 = 16384
B) 256 × 64 + 64 = 16448
C) 256 × 64 + 256 = 16640
D) 256 + 64 = 320

**答案**: B
**解析**: weight 矩阵 256×64 = 16384 个参数 + bias 向量 64 个参数 = 16448。

### Q4 (选择题 · 10分) [Training]
训练循环中，`optimizer.zero_grad()` 应该放在什么位置？
A) 在 loss.backward() 之后
B) 在 optimizer.step() 之后
C) 在 loss.backward() 之前
D) 在 model(data) 之前，放哪里都行

**答案**: C（也可接受 D）
**解析**: zero_grad() 必须在 backward() 之前调用，否则新梯度会和旧梯度累加导致错误更新。标准位置是在前向传播之后、backward 之前，或者每次迭代的最开始。

### Q5 (判断题 · 10分) [Tensor]
`torch.zeros(3, 4)` 和 `torch.zeros((3, 4))` 创建的张量是一样的。(T/F)

**答案**: T
**解析**: PyTorch 支持两种传参方式：直接传多个整数或传一个元组，结果都是 shape (3, 4) 的全零张量。

### Q6 (判断题 · 10分) [Autograd]
在 `with torch.no_grad():` 代码块中创建的张量，即使设置了 `requires_grad=True`，也不会被追踪梯度。(T/F)

**答案**: T
**解析**: `torch.no_grad()` 上下文管理器会禁用梯度追踪。在其中进行的运算不会构建计算图，即使输入张量有 `requires_grad=True`，输出也不会有 grad_fn。

### Q7 (代码题 · 10分) [Tensor + Autograd]
创建一个 requires_grad=True 的张量 `w = [1.0, 2.0, 3.0]`，计算 `loss = (w ** 2).sum()`，然后求 w 的梯度。写出完整代码和预期输出。

**参考答案**:
```python
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (w ** 2).sum()
loss.backward()
print(w.grad)  # tensor([2., 4., 6.])  因为 d(w²)/dw = 2w
```
**评分标准**: 代码正确 6分，预期输出正确并能解释原因 4分。

### Q8 (代码题 · 10分) [nn.Module + Training]
补全以下模型定义，使其成为一个 3 层网络（输入 784 → 隐藏 256 → 隐藏 128 → 输出 10），每个隐藏层后面加 ReLU 激活。

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 请补全
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
```

**参考答案**:
```python
self.net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```
**评分标准**: 结构完全正确 7分，激活函数位置正确 3分。最后一层不应加 ReLU（因为后面接 CrossEntropyLoss 已内含 Softmax）。

### Q9 (简答题 · 10分) [Training]
请描述一个完整的训练循环（一个 iteration）的 5 个步骤，并解释为什么顺序不能随意调换。

**参考答案**:
1. **前向传播** `output = model(data)` — 计算预测值
2. **计算损失** `loss = criterion(output, target)` — 衡量预测与真实的差距
3. **清零梯度** `optimizer.zero_grad()` — 防止梯度累加
4. **反向传播** `loss.backward()` — 计算各参数的梯度
5. **更新参数** `optimizer.step()` — 用梯度更新权重

顺序原因：必须先有 loss 才能反向传播，必须先反向传播才能用梯度更新参数，必须在反向传播前清零否则梯度会累加。

**评分标准**: 5 步都写对 6分，能解释顺序原因 4分。

### Q10 (思考题 · 10分) [综合]
你训练了一个 MNIST 分类器，训练集准确率 99.5%，测试集准确率 85%。这说明了什么问题？你会怎么改善？

**参考答案**:
这是典型的**过拟合**——模型在训练集上"背答案"，泛化能力差。

改善方法（答出任意 3 点即满分）：
- 增加数据量或使用数据增强（Data Augmentation）
- 添加正则化：Dropout、L2 正则化（weight_decay）
- 简化模型结构（减少参数量）
- 使用 Early Stopping（验证集 loss 不再下降时停止训练）
- 添加 BatchNorm

**评分标准**: 识别出过拟合 4分，给出合理改善方案 6分。

## 薄弱点诊断

根据答题情况，标记学生在以下知识模块的掌握程度：

| 模块 | 对应题目 | 掌握程度 |
|------|---------|---------|
| Tensor 基本操作 | Q1, Q5 | |
| Autograd 求导 | Q2, Q6, Q7 | |
| nn.Module 模型搭建 | Q3, Q8 | |
| 训练循环 | Q4, Q9 | |
| 综合理解（过拟合等） | Q10 | |
