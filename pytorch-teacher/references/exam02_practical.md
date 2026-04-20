# 阶段考试 2：实战进阶篇 (覆盖 Lesson 5-7)

## 考试说明
- 共 10 道题，满分 100 分
- 题型：选择题 × 4、判断题 × 2、代码题 × 2、思考/简答题 × 2
- 评分标准：90+ 优秀 🌟 | 70-89 良好 👍 | 60-69 及格 ✅ | <60 需复习 📖

## 考题

### Q1 (选择题 · 10分) [Dataset]
自定义 Dataset 类必须实现哪些方法？
A) `__init__` 和 `__call__`
B) `__len__` 和 `__getitem__`
C) `__iter__` 和 `__next__`
D) `forward` 和 `backward`

**答案**: B
**解析**: `__len__` 返回数据集大小，`__getitem__` 根据索引返回一条数据。DataLoader 依赖这两个方法来工作。`__init__` 虽然通常也写，但不是抽象方法的强制要求。

### Q2 (选择题 · 10分) [DataLoader]
数据集有 200 个样本，batch_size=64, drop_last=False，一个 epoch 有几个 batch？
A) 3
B) 4
C) 3.125
D) 200

**答案**: B
**解析**: 200 ÷ 64 = 3 余 8。drop_last=False（默认值）保留最后一个不完整的 batch（只有 8 个样本），所以共 4 个 batch。

### Q3 (选择题 · 10分) [CNN]
输入 shape 为 [batch, 1, 28, 28]，经过 `Conv2d(1, 32, kernel_size=5, padding=0)` 后的输出 shape 是？
A) [batch, 32, 28, 28]
B) [batch, 32, 24, 24]
C) [batch, 32, 14, 14]
D) [batch, 1, 24, 24]

**答案**: B
**解析**: 输出尺寸 = (输入尺寸 - kernel_size + 2×padding) / stride + 1 = (28 - 5 + 0) / 1 + 1 = 24。通道从 1 变为 32。

### Q4 (选择题 · 10分) [Sequence]
`nn.Embedding(5000, 128)` 的参数矩阵有多少个可训练参数？
A) 5000
B) 128
C) 5000 + 128 = 5128
D) 5000 × 128 = 640000

**答案**: D
**解析**: Embedding 层本质是一个 5000×128 的查找表（lookup table），每一行是一个词的 128 维向量表示，所有元素都是可训练参数。

### Q5 (判断题 · 10分) [DataLoader]
DataLoader 的 shuffle=True 在训练和测试时都应该开启，以保证数据的随机性。(T/F)

**答案**: F
**解析**: shuffle=True 只在训练时使用（打乱顺序防止模型学到数据排列模式）。测试/验证时应关闭 shuffle，以确保结果可复现。

### Q6 (判断题 · 10分) [CNN]
MaxPool2d(2) 会使特征图的宽和高各减半，但不会改变通道数。(T/F)

**答案**: T
**解析**: MaxPool2d(2) 在每个 2×2 区域内取最大值，所以宽高各缩小一半。它在每个通道上独立操作，不改变通道数。

### Q7 (代码题 · 10分) [Dataset]
写一个自定义 Dataset，生成 y = 3x + 5 + 噪声 的数据（共 500 个样本），返回 (x, y) 对。

**参考答案**:
```python
from torch.utils.data import Dataset

class LinearDataset(Dataset):
    def __init__(self, num_samples=500):
        self.x = torch.rand(num_samples, 1) * 10     # x ∈ [0, 10)
        self.y = 3 * self.x + 5 + torch.randn(num_samples, 1) * 0.5  # 加噪声

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```
**评分标准**: 继承 Dataset 2分，`__len__` 正确 2分，`__getitem__` 正确 3分，数据生成合理 3分。

### Q8 (代码题 · 10分) [CNN]
写出一个简单 CNN 的 `__init__` 方法，要求：
- 输入：单通道 28×28 图像
- 两个卷积层：Conv(1→16, k=3, p=1) + ReLU + MaxPool(2)，Conv(16→32, k=3, p=1) + ReLU + MaxPool(2)
- 全连接层：展平 → 输出 10 类

**参考答案**:
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),   # → [16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → [16, 14, 14]
            nn.Conv2d(16, 32, 3, padding=1),   # → [32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → [32, 7, 7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # → [32*7*7] = [1568]
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```
**评分标准**: 卷积层参数正确 3分，shape 推导正确 3分，全连接输入维度正确 2分，整体结构完整 2分。

### Q9 (简答题 · 10分) [Data Augmentation]
什么是数据增强（Data Augmentation）？它为什么能提高模型性能？列举至少 3 种常用的图像数据增强方法。

**参考答案**:
数据增强是在不收集新数据的情况下，通过对现有训练数据做随机变换来扩充数据多样性的技术。

为什么有效：让模型见到更多变化，学到更鲁棒(robust)的特征，减少过拟合。

常用方法：
1. 随机水平翻转 (RandomHorizontalFlip)
2. 随机旋转 (RandomRotation)
3. 随机裁剪 (RandomCrop / RandomResizedCrop)
4. 颜色抖动 (ColorJitter: 亮度、对比度、饱和度)
5. 随机擦除 (RandomErasing)

**评分标准**: 解释概念 3分，解释原因 3分，列举方法 4分（每个方法1分，最多4分）。

### Q10 (思考题 · 10分) [综合]
RNN 处理长序列时会遇到什么问题？LSTM 是怎么解决的？为什么后来的 Transformer 又抛弃了 RNN 的结构？

**参考答案**:
- **RNN 的问题**: 梯度消失/爆炸。信息必须逐步传递，长序列中早期信息逐渐"稀释"。
- **LSTM 的解决方案**: 引入门控机制（遗忘门、输入门、输出门）和细胞状态(cell state)。细胞状态像传送带，信息可以直接通过，缓解了梯度消失。但本质上仍然是顺序处理，无法并行。
- **Transformer 的优势**: 用 Self-Attention 替代循环结构，每个位置可以直接注意到所有其他位置（O(1) 步），而不是像 RNN 要经过 N 步传递。Attention 天然支持并行计算，在 GPU 上训练效率远高于 RNN。

**评分标准**: RNN 问题描述 3分，LSTM 解决方案 3分，Transformer 优势 4分。

## 薄弱点诊断

| 模块 | 对应题目 | 掌握程度 |
|------|---------|---------|
| Dataset 自定义 | Q1, Q7 | |
| DataLoader 配置 | Q2, Q5 | |
| 数据增强 | Q9 | |
| CNN 结构与 shape 推导 | Q3, Q6, Q8 | |
| 序列模型与 Embedding | Q4, Q10 | |
