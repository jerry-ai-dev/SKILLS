# Lesson 6: 卷积神经网络 CNN

## 教学目标
理解卷积操作的原理，用 CNN 做图像分类。

## 讲解要点

### 1. 卷积的直觉
- 全连接层：每个像素独立看 → 效率低，忽略空间关系
- 卷积层：用"滑动窗口"提取局部特征 → 高效，保留空间结构
- 浅层学边缘 → 中层学纹理 → 深层学物体部件

### 2. 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# ========== 理解卷积操作 ==========
# Conv2d(输入通道, 输出通道, 卷积核大小)
conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
# 输入: [batch=1, channels=1, height=28, width=28]
x = torch.randn(1, 1, 28, 28)
out = conv(x)
print(f"卷积: {x.shape} → {out.shape}")
# 输出: [1, 16, 28, 28]（padding=1保持尺寸不变）

# 池化层: 缩小特征图
pool = nn.MaxPool2d(2)
out_pooled = pool(out)
print(f"池化: {out.shape} → {out_pooled.shape}")
# 输出: [1, 16, 14, 14]（尺寸减半）

# ========== CNN 分类器 ==========
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 1 → 32 通道
            nn.Conv2d(1, 32, 3, padding=1),  # [B,32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                  # [B,32,14,14]
            
            # Block 2: 32 → 64 通道
            nn.Conv2d(32, 64, 3, padding=1),  # [B,64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                  # [B,64,7,7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # [B, 64*7*7]
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),                   # 防过拟合
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN()
print(f"\nCNN 结构:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {total_params:,}")

# ========== 训练 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # 评估
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            pred = model(data).argmax(1)
            correct += (pred == target).sum().item()
    
    acc = correct / len(test_data)
    print(f"Epoch {epoch+1}: Test Accuracy = {acc:.2%}")
```

## 对比：全连接 vs CNN

| 特性 | 全连接 (Lesson 4) | CNN |
|------|-------------------|-----|
| MNIST 准确率 | ~97% | ~99% |
| 参数量 | ~100K | ~90K |
| 理解空间结构 | ❌ | ✅ |
| 平移不变性 | ❌ | ✅ |

## 测验题目

### Q1
输入 [1, 3, 32, 32]，经过 Conv2d(3, 16, 3, padding=1) 后形状是？

**答案**: [1, 16, 32, 32] — padding=1 保持宽高不变，通道从 3 变为 16

### Q2
MaxPool2d(2) 的作用是？为什么要用它？

**答案**: 将特征图宽高缩小一半。作用：减少计算量，增大感受野，提供一定的平移不变性。

### Q3 (选择题)
Conv2d(3, 16, kernel_size=3) 的可训练参数数量是？
A) 3×16 = 48  B) 3×16×3×3 = 432  C) 3×16×3×3 + 16 = 448  D) 16×3×3 = 144

**答案**: C — 每个卷积核大小为 3×3×3（宽×高×输入通道），共 16 个卷积核，参数量 = 3×3×3×16 = 432（权重）+ 16（偏置）= **448**。

### Q4 (判断题)
CNN 相比全连接网络的两大核心优势是"参数共享"和"局部连接"。(T/F)

**答案**: T — 卷积核在整张图上滑动共享参数（参数共享），每个输出只与输入的局部区域相连（局部连接）。这两点使得 CNN 参数量远小于全连接网络，同时更适合捕捉空间特征。

### Q5 (思考题)
为什么 CNN 通常采用"通道数逐层增加，空间尺寸逐层减小"的设计模式？

**答案**: 浅层用少量通道提取边缘、纹理等低级特征，空间分辨率高以保留位置信息。深层用更多通道组合出复杂的高级特征（如眼睛、轮子），同时通过池化缩小空间尺寸以减少计算量并增大感受野。这种设计使网络能从简单到复杂逐步理解图片内容。

## 实践任务
1. 将模型修改为适用于 CIFAR-10（3通道彩色32×32图片，10类）
2. 加入 BatchNorm 层，观察训练速度是否加快
3. 尝试加入残差连接（skip connection）
