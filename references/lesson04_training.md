# Lesson 4: 训练循环 Training Loop

## 教学目标
掌握完整的训练流程：前向传播 → 计算损失 → 反向传播 → 更新参数。

## 讲解要点

### 1. 训练的核心流程
```
数据 → 模型预测 → 计算差距(loss) → 算梯度(backward) → 调参数(step) → 循环
```

### 2. 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ========== 准备数据 ==========
transform = transforms.Compose([
    transforms.ToTensor(),          # 像素值 → 0~1 张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

# 下载 MNIST 数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# ========== 定义模型 ==========
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),          # 28×28 → 784
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return self.net(x)

model = DigitClassifier()

# ========== 定义损失函数和优化器 ==========
criterion = nn.CrossEntropyLoss()  # 分类任务用交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== 训练循环 ==========
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()  # 训练模式
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        # 1. 前向传播
        output = model(data)
        
        # 2. 计算损失
        loss = criterion(output, target)
        
        # 3. 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        
        # 4. 更新参数
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total

# ========== 评估函数 ==========
def evaluate(model, loader):
    model.eval()  # 评估模式
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不需要梯度
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total

# ========== 开始训练 ==========
epochs = 5
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")

# ========== 保存模型 ==========
torch.save(model.state_dict(), 'mnist_model.pth')
print("模型已保存！")
```

## 关键概念速查

| 概念 | 作用 | 常用选择 |
|------|------|----------|
| Loss 函数 | 衡量预测与真实的差距 | 分类→CrossEntropy, 回归→MSE |
| 优化器 | 根据梯度更新参数 | Adam(通用), SGD(经典) |
| 学习率 | 每步调整的幅度 | 0.001 是常见起点 |
| Epoch | 完整遍历一次数据 | 通常 5~100 |
| Batch | 每次用多少样本 | 32, 64, 128 |

## 测验题目

### Q1 (排序题)
将训练循环的步骤排列正确顺序:
a) loss.backward()  b) optimizer.step()  c) output = model(data)  d) optimizer.zero_grad()  e) loss = criterion(output, target)

**答案**: c → e → d → a → b（前向 → 算loss → 清梯度 → 反向 → 更新）

### Q2 (选择题)
model.eval() 和 model.train() 的区别主要影响:
A) 学习率  B) Dropout 和 BatchNorm  C) 损失函数  D) 数据集

**答案**: B

### Q3 (思考题)
为什么要把数据分成 train 和 test？能不能只用 train 数据？

**答案**: 防止过拟合。模型可能只是"背答案"而不是"学规律"。test 数据从未参与训练，能真实反映模型的泛化能力。

### Q4 (选择题)
在评估模型时用 `torch.no_grad()` 主要是为了：
A) 防止模型参数被修改  B) 提高预测准确率  C) 不构建计算图，节省内存和加速  D) 切换到 CPU 运算

**答案**: C — 评估时不需要反向传播，`torch.no_grad()` 关闭梯度追踪可以减少约 50% 内存占用并提升推理速度。

### Q5 (代码题)
模型输出 logits 的 shape 是 `[64, 10]`（batch_size=64, 10个类别），写出将 logits 转为预测类别标签的代码。

**参考答案**:
```python
pred = logits.argmax(dim=1)  # shape: [64]，每个样本取概率最大的类别
```
关键是 `dim=1` 表示在类别维度上取最大值的索引。

## 实践任务
1. 在上述代码基础上，尝试修改网络结构（加深/加宽），观察准确率变化
2. 尝试不同的学习率 (0.1, 0.01, 0.001, 0.0001)，观察训练效果
3. 在训练中途保存和加载模型
