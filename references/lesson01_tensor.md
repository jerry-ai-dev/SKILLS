# Lesson 1: Tensor 张量入门

## 教学目标
学生能理解张量的概念，能用 PyTorch 创建和操作张量。

## 讲解要点

### 1. 什么是张量
- 标量(0维) → 向量(1维) → 矩阵(2维) → 张量(N维)
- 张量就是"多维数组"，是深度学习的基本数据结构
- 类似 NumPy 的 ndarray，但能在 GPU 上运算

### 2. 代码示例

```python
import torch

# ========== 创建张量 ==========
# 从列表创建
a = torch.tensor([1, 2, 3])
print(f"向量: {a}, 形状: {a.shape}, 类型: {a.dtype}")

# 创建矩阵
b = torch.tensor([[1, 2], [3, 4]])
print(f"矩阵:\n{b}, 形状: {b.shape}")

# 常用创建方法
zeros = torch.zeros(3, 4)       # 全0
ones = torch.ones(2, 3)         # 全1
rand = torch.rand(2, 3)         # 0~1随机
randn = torch.randn(2, 3)      # 标准正态分布
arange = torch.arange(0, 10, 2) # 等差序列

print(f"随机张量:\n{rand}")

# ========== 基本运算 ==========
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 逐元素运算
print(f"加法: {x + y}")
print(f"乘法: {x * y}")
print(f"求和: {x.sum()}")
print(f"均值: {x.mean()}")

# 矩阵乘法
A = torch.rand(2, 3)
B = torch.rand(3, 4)
C = A @ B  # 或 torch.matmul(A, B)
print(f"矩阵乘法: {A.shape} @ {B.shape} = {C.shape}")

# ========== 形状操作 ==========
t = torch.arange(12)
print(f"原始: {t.shape}")
t_reshaped = t.reshape(3, 4)
print(f"reshape 后: {t_reshaped.shape}")
print(t_reshaped)

# ========== GPU (如果可用) ==========
if torch.cuda.is_available():
    device = torch.device('cuda')
    x_gpu = x.to(device)
    print(f"GPU 上的张量: {x_gpu.device}")
else:
    print("没有 GPU，使用 CPU 也完全可以学习！")
```

## 测验题目

### Q1 (选择题)
`torch.randn(3, 4)` 创建了什么形状的张量？
A) 3个元素的向量  B) 4×3的矩阵  C) 3×4的矩阵  D) 12个元素的向量

**答案**: C

### Q2 (代码题)
创建两个 2×3 的随机张量，计算它们的逐元素乘积和矩阵乘积（提示：第二个需要转置）。

**参考答案**:
```python
a = torch.rand(2, 3)
b = torch.rand(2, 3)
# 逐元素乘积
print(a * b)
# 矩阵乘积 (需要转置 b)
print(a @ b.T)  # (2,3) @ (3,2) = (2,2)
```

### Q3 (思考题)
为什么深度学习要用张量而不是普通的 Python 列表？

**答案要点**: 
- 张量支持 GPU 加速，Python 列表不行
- 张量运算是向量化的（底层 C++ 实现），比 Python 循环快几百倍
- 张量自动支持求导（autograd），这是训练神经网络的基础

### Q4 (判断题)
`torch.tensor([1, 2, 3])` 创建的张量默认 dtype 是 float32。(T/F)

**答案**: F — 整数输入默认创建 int64 (torch.long) 类型。要得到 float32 需要写 `torch.tensor([1.0, 2.0, 3.0])` 或 `torch.tensor([1, 2, 3], dtype=torch.float32)`。

### Q5 (代码题)
创建一个 shape 为 (2, 3, 4) 的随机张量，然后：① 用 `view` 把它 reshape 成 (6, 4)；② 用 `permute` 交换第 0 维和第 2 维，写出结果的 shape。

**参考答案**:
```python
t = torch.randn(2, 3, 4)
print(t.view(6, 4).shape)       # torch.Size([6, 4])
print(t.permute(2, 1, 0).shape) # torch.Size([4, 3, 2])
```

## 实践任务
创建一个代表 3×3 灰度图片的张量（像素值 0~255），然后:
1. 将像素值归一化到 0~1
2. 增加亮度（每个像素 +0.2，但不超过 1.0）
3. 打印处理前后的张量
