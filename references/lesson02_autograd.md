# Lesson 2: 自动求导 Autograd

## 教学目标
理解梯度和反向传播的概念，能用 autograd 自动计算梯度。

## 讲解要点

### 1. 为什么需要求导
- 训练神经网络的核心：找到让 loss 最小的参数
- 梯度 = loss 对参数的偏导数 = "参数该往哪个方向调整"
- 梯度下降：参数 = 参数 - 学习率 × 梯度

### 2. 代码示例

```python
import torch

# ========== 基本 autograd ==========
# 创建需要求导的张量
x = torch.tensor(3.0, requires_grad=True)
# 定义一个函数 y = x^2 + 2x + 1
y = x ** 2 + 2 * x + 1
print(f"x = {x.item()}, y = {y.item()}")

# 反向传播，计算梯度
y.backward()
# dy/dx = 2x + 2, 当 x=3 时 = 8
print(f"dy/dx = {x.grad.item()}")  # 输出 8.0

# ========== 梯度下降直觉 ==========
# 目标：找到 y = (x-5)^2 的最小值点
x = torch.tensor(0.0, requires_grad=True)
learning_rate = 0.1

for step in range(20):
    y = (x - 5) ** 2  # 最小值在 x=5
    y.backward()
    
    # 手动梯度下降（暂不用优化器）
    with torch.no_grad():
        x -= learning_rate * x.grad
    x.grad.zero_()  # 清零梯度！
    
    if step % 5 == 0:
        print(f"Step {step}: x = {x.item():.4f}, y = {y.item():.4f}")

print(f"最终 x ≈ {x.item():.4f}（应接近 5.0）")

# ========== 手动实现线性回归 ==========
# 生成数据: y = 3x + 1 + 噪声
torch.manual_seed(42)
X = torch.rand(100, 1) * 10  # 100个样本
y_true = 3 * X + 1 + torch.randn(100, 1) * 0.5

# 初始化参数
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01

for epoch in range(100):
    # 前向传播
    y_pred = w * X + b
    
    # 计算损失 (MSE)
    loss = ((y_pred - y_true) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    
    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\n学到的参数: y = {w.item():.2f}x + {b.item():.2f}")
print(f"真实参数:    y = 3.00x + 1.00")
```

## 测验题目

### Q1 (选择题)
调用 `y.backward()` 后，梯度存储在哪里？
A) y.grad  B) x.grad  C) torch.grad  D) backward.grad

**答案**: B — 梯度存储在叶子节点（requires_grad=True 的变量）的 .grad 属性中

### Q2 (判断题)
每次调用 backward() 前需要清零梯度，否则梯度会累加。(T/F)

**答案**: T — PyTorch 默认累加梯度，需要手动 .grad.zero_() 或 optimizer.zero_grad()

### Q3 (代码题)
用 autograd 验证：当 $f(x) = x^3$ 时，$x = 2$ 处的梯度是 12。

**参考答案**:
```python
x = torch.tensor(2.0, requires_grad=True)
f = x ** 3
f.backward()
print(x.grad)  # tensor(12.)
```

### Q4 (选择题)
`with torch.no_grad():` 代码块的作用是？
A) 删除所有梯度  B) 暂停梯度计算，节省内存  C) 将梯度全部清零  D) 关闭 GPU 加速

**答案**: B — 在该上下文中 PyTorch 不再构建计算图、不追踪梯度，适用于推理阶段或不需要求导的场景，可以节省内存和加速。

### Q5 (代码题)
用 autograd 验证：$y = \sin(x)$ 在 $x = \pi/4$ 处的梯度等于 $\cos(\pi/4) \approx 0.7071$。

**参考答案**:
```python
import math
x = torch.tensor(math.pi / 4, requires_grad=True)
y = torch.sin(x)
y.backward()
print(x.grad)           # tensor(0.7071)
print(math.cos(math.pi / 4))  # 0.7071...
```

## 实践任务
修改线性回归例子:
1. 改为学习 y = 2x² + 3x - 1 的参数（需要增加一个参数）
2. 画出 loss 的变化曲线（用 print 或 matplotlib）
3. 尝试不同学习率，观察收敛速度
