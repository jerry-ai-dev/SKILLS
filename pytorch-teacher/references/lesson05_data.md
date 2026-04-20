# Lesson 5: 数据加载 Dataset & DataLoader

## 教学目标
学会自定义 Dataset，用 DataLoader 高效加载数据。

## 讲解要点

### 1. Dataset 和 DataLoader 的关系
- Dataset: "有什么数据"（定义如何获取一条数据）
- DataLoader: "怎么喂数据"（批量、打乱、多进程）

### 2. 代码示例

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ========== 自定义 Dataset ==========
class SimpleDataset(Dataset):
    """一个简单的数据集：y = 2x + 1 + 噪声"""
    def __init__(self, num_samples=1000):
        self.x = torch.rand(num_samples, 1) * 10
        self.y = 2 * self.x + 1 + torch.randn(num_samples, 1) * 0.5
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 创建数据集
dataset = SimpleDataset(1000)
print(f"数据集大小: {len(dataset)}")
print(f"第一条数据: x={dataset[0][0].item():.2f}, y={dataset[0][1].item():.2f}")

# ========== DataLoader ==========
loader = DataLoader(
    dataset,
    batch_size=32,      # 每批32个样本
    shuffle=True,       # 每个epoch随机打乱
    num_workers=0,      # Windows 下建议用 0
    drop_last=False,    # 是否丢弃最后不足一批的数据
)

# 遍历一个epoch
for batch_idx, (x_batch, y_batch) in enumerate(loader):
    if batch_idx == 0:
        print(f"\n第一批数据: x形状={x_batch.shape}, y形状={y_batch.shape}")
    if batch_idx >= 2:
        break
print(f"总共 {len(loader)} 个batch")

# ========== 数据集分割 ==========
from torch.utils.data import random_split

full_dataset = SimpleDataset(1000)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
print(f"\n训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")

# ========== 图像数据增强 ==========
from torchvision import transforms

# 常用的数据增强组合
image_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),    # 随机水平翻转
    transforms.RandomRotation(10),        # 随机旋转 ±10度  
    transforms.ColorJitter(brightness=0.2),  # 随机亮度
    transforms.ToTensor(),                # 转为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准化
                        std=[0.229, 0.224, 0.225]),
])
print("\n数据增强 pipeline 已准备好")
print("包含: 随机翻转 → 随机旋转 → 亮度调整 → 转张量 → 标准化")
```

## 测验题目

### Q1
自定义 Dataset 需要实现哪两个方法？

**答案**: `__len__` (返回数据集大小) 和 `__getitem__` (返回第 idx 条数据)

### Q2
DataLoader 的 shuffle=True 在什么时候有用？

**答案**: 训练时有用（打乱数据顺序，防止模型学到数据的排列模式）。测试时通常不需要 shuffle。

### Q3 (代码题)
写一个 Dataset，加载 CSV 文件中的数据（前 N-1 列是特征，最后一列是标签）。

**参考答案**:
```python
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

### Q4 (选择题)
数据集有 100 个样本，DataLoader 设置 batch_size=32, drop_last=True，每个 epoch 有几个 batch？
A) 3  B) 4  C) 3.125  D) 100

**答案**: A — 100 ÷ 32 = 3 余 4。drop_last=True 会丢弃最后不足一个 batch 的 4 个样本，所以只有 3 个 batch（共 96 个样本被使用）。

### Q5 (判断题)
`random_split` 分割数据集时会复制一份数据到新的数据集对象中。(T/F)

**答案**: F — `random_split` 返回的是 `Subset` 对象，只保存了索引列表，底层仍然共享同一份原始数据，不会复制数据，所以不会额外占用内存。

## 实践任务
1. 创建一个 Dataset，生成螺旋形数据（2类或3类的螺旋点集）
2. 用 DataLoader 加载，训练一个分类网络
3. 可视化数据分布和决策边界
