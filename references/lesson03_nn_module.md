# Lesson 3: nn.Module 搭建神经网络

## 教学目标
学会用 nn.Module 定义神经网络，理解层、激活函数和前向传播。

## 讲解要点

### 1. 为什么用 nn.Module
- 上一课手动管理 w、b 太麻烦
- nn.Module 自动管理参数、支持保存/加载、方便组合

### 2. 代码示例

```python
import torch
import torch.nn as nn

# ========== 最简单的网络 ==========
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 1个输入，1个输出
    
    def forward(self, x):
        return self.linear(x)

model = SimpleNet()
print("模型结构:")
print(model)
print(f"\n参数: {list(model.parameters())}")

# 测试前向传播
x = torch.tensor([[1.0], [2.0], [3.0]])
output = model(x)  # 调用 model 就等于调用 forward
print(f"输入: {x.T}")
print(f"输出: {output.T}")

# ========== 多层网络 (MNIST 手写数字) ==========
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST 图片是 28×28 = 784 像素
        self.net = nn.Sequential(
            nn.Linear(784, 128),   # 第一层: 784 → 128
            nn.ReLU(),             # 激活函数
            nn.Linear(128, 64),    # 第二层: 128 → 64
            nn.ReLU(),
            nn.Linear(64, 10),     # 输出层: 64 → 10 (0-9 十个数字)
        )
    
    def forward(self, x):
        return self.net(x)

model = DigitClassifier()
print("\n手写数字分类器:")
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# 模拟输入（1张图片，展平为784维向量）
fake_image = torch.randn(1, 784)
output = model(fake_image)
print(f"\n输入形状: {fake_image.shape}")
print(f"输出形状: {output.shape}")
print(f"输出（各数字的得分）: {output}")

# 预测的数字
pred = output.argmax(dim=1)
print(f"预测数字: {pred.item()}")

# ========== 理解激活函数 ==========
print("\n===== 激活函数对比 =====")
x = torch.linspace(-5, 5, 11)
print(f"输入:    {x.tolist()}")
print(f"ReLU:    {torch.relu(x).tolist()}")
print(f"Sigmoid: {torch.sigmoid(x).tolist()}")
print(f"Tanh:    {torch.tanh(x).tolist()}")
# ReLU: 负数变0，正数不变 —— 简单高效，是最常用的激活函数
```

## 测验题目

### Q1 (选择题)
nn.Linear(784, 128) 有多少个参数？
A) 784  B) 128  C) 784×128  D) 784×128 + 128

**答案**: D — weight 矩阵 784×128 = 100352，加上 bias 128 = 100480

### Q2 (填空)
ReLU 激活函数的公式是 _____，它的作用是引入 _____。

**答案**: max(0, x)，非线性

### Q3 (思考题)
如果去掉所有激活函数，多层 Linear 层叠在一起等价于什么？

**答案**: 等价于一个 Linear 层。因为多个线性变换的组合仍然是线性变换。这就是为什么需要激活函数——引入非线性才能学习复杂模式。

### Q4 (判断题)
`nn.Sequential` 中的层会严格按照添加顺序依次执行前向传播。(T/F)

**答案**: T — `nn.Sequential` 会按照构造时的顺序依次调用每一层的 `forward` 方法，上一层的输出自动作为下一层的输入。

### Q5 (代码题)
给定以下模型，手动计算总参数量：
```python
nn.Sequential(
    nn.Linear(10, 32),   # ?
    nn.ReLU(),
    nn.Linear(32, 5),    # ?
)
```

**答案**: 第一层参数 = 10×32 + 32 = 352，第二层参数 = 32×5 + 5 = 165，ReLU 无参数。总计 352 + 165 = **517**。

## 实践任务
1. 搭建一个 3 层的网络：输入 2 维 → 隐藏 16 维 → 隐藏 8 维 → 输出 1 维
2. 用这个网络拟合 XOR 问题（输入 [0,0],[0,1],[1,0],[1,1]，输出 [0,1,1,0]）
3. 打印每一层的参数形状
