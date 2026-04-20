# Lesson 8: 梯度累积 & 混合精度训练

## 教学目标
理解梯度累积和 bf16 的原理，掌握在 PyTorch 中的使用方法，能估算训练时的显存占用。

---

## 讲解要点

### 1. 为什么需要梯度累积？

**问题**：大模型训练效果依赖大 batch size（通常需要 batch=256+），但显存放不下这么大的 batch。

**梯度累积（Gradient Accumulation）**：
- 把大 batch 拆成 $K$ 个小步
- 每步只做前向 + 反向传播，**但不更新参数**（不调用 `optimizer.step()`）
- 累积 $K$ 步的梯度后，统一更新一次

效果等价于 `effective_batch_size = per_device_batch × accumulation_steps`

$$\text{Effective Batch} = N_{\text{devices}} \times B_{\text{device}} \times K_{\text{accum}}$$

**注意**：accumulation 期间的 loss 要除以 $K$（归一化），否则梯度是 $K$ 倍大：

```python
loss = loss / accumulation_steps  # ← 关键！
loss.backward()
```

---

### 2. 混合精度训练（Mixed Precision）

**fp32（单精度）**：4 bytes，范围 ±3.4×10³⁸，精度 ~7 位十进制
**fp16（半精度）**：2 bytes，范围 ±6.5×10⁴，精度 ~3 位十进制
**bf16（Brain Float 16）**：2 bytes，范围同 fp32（±3.4×10³⁸），精度较低

**混合精度策略**：
- **前向传播 + 反向传播**：用 fp16/bf16（节省显存，加速）
- **参数更新（梯度）**：用 fp32（防止数值不稳定）
- **参数存储（Master weights）**：用 fp32（保持精度）

---

### 3. fp16 vs bf16：大模型选哪个？

| | fp16 | bf16 |
|---|------|------|
| 数值范围 | 小（±65504） | 大（同 fp32） |
| 数值精度 | 较高 | 较低 |
| Overflow 风险 | ⚠️ 高（需要 Loss Scaling） | ✅ 低 |
| 适合场景 | 推理、小模型 | **大模型训练（推荐）** |
| 硬件支持 | 所有 GPU | A100, H100, 4090 等新卡 |

**结论**：大模型训练（7B+）强烈推荐 **bf16**，因为数值范围等同 fp32，不会 overflow，不需要 Loss Scaling。

---

### 4. Loss Scaling（fp16 专用）

fp16 的梯度可能变成 0（underflow），Loss Scaling 的解决方案：

1. 将 loss 乘以一个大数 $S$（如 $S = 2^{12}$）
2. 反向传播，梯度也被放大 $S$ 倍（不 underflow 了）
3. 参数更新前，把梯度除以 $S$，恢复正确值
4. 如果梯度溢出（inf/nan），跳过这步更新，减小 $S$

PyTorch 的 `torch.cuda.amp.GradScaler` 自动完成这个过程。

---

### 5. 显存估算

训练时显存占用的粗略估算（bf16 混合精度）：

| 内容 | 大小 |
|------|------|
| 模型参数（bf16） | `params × 2 bytes` |
| 梯度（fp32） | `params × 4 bytes` |
| 优化器状态（Adam, fp32） | `params × 8 bytes` |
| 激活值（前向传播缓存） | 取决于 batch size 和 seq_len |

总计约 `params × 14-16 bytes`（不含激活）

例如：7B 模型 ≈ 7×10⁹ × 14 bytes ≈ **98 GB**（所以需要多卡）

---

## 代码示例

```python
import torch
import torch.nn as nn
import time

# ===== 梯度累积实现 =====

print("==== 梯度累积演示 ====\n")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model     = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 目标：等效 batch_size = 32，但每步只处理 8 个样本
total_data      = 64
mini_batch_size = 8
accumulation_k  = total_data // mini_batch_size  # = 8 步累积

X = torch.randn(total_data, 64)
Y = torch.randn(total_data)

# ===== 方式1：真实大 batch（需要大显存，用于对比） =====
optimizer.zero_grad()
outputs = model(X)
loss_big = criterion(outputs, Y)
loss_big.backward()
param_grad_big = model.net[0].weight.grad.clone()
optimizer.step()
print(f"真实大 batch loss: {loss_big.item():.6f}")

# ===== 方式2：梯度累积（等效大 batch） =====
model2 = SimpleModel()  # 重置模型，用相同初始参数
model2.load_state_dict(model.state_dict())
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
optimizer2.zero_grad()

accumulated_loss = 0.0
for step in range(accumulation_k):
    x_mini = X[step * mini_batch_size : (step + 1) * mini_batch_size]
    y_mini = Y[step * mini_batch_size : (step + 1) * mini_batch_size]

    outputs_mini = model2(x_mini)
    loss_mini    = criterion(outputs_mini, y_mini)

    # 关键：归一化 loss（否则梯度是 K 倍大）
    loss_mini = loss_mini / accumulation_k
    loss_mini.backward()  # 梯度在 backward() 间自动累加

    accumulated_loss += loss_mini.item()

# 累积 K 步后统一更新
optimizer2.step()
print(f"梯度累积等效 loss: {accumulated_loss:.6f}")


# ===== 混合精度训练（bf16） =====
print("\n==== 混合精度训练演示 ====\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("（无 GPU，仅展示代码结构；autocast 在 CPU 上仍可运行）")

model3 = SimpleModel().to(device)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
X_dev = X.to(device)
Y_dev = Y.to(device)

# bf16 混合精度（推荐大模型使用）
dtype = torch.bfloat16 if device == "cuda" else torch.float32

with torch.autocast(device_type=device, dtype=dtype):
    outputs3 = model3(X_dev)
    loss3    = criterion(outputs3, Y_dev)

print(f"混合精度 loss: {loss3.item():.6f}")
print(f"自动转换的数据类型: {outputs3.dtype}")

# 注意：bf16 时不需要 GradScaler！（只有 fp16 才需要）
loss3.backward()
optimizer3.step()
print("bf16 训练步骤完成（无需 GradScaler）✅")


# ===== 显存估算 =====
print("\n==== 显存估算 ====")
model_sizes = {"GPT-2 (117M)": 117e6, "LLaMA-2 7B": 7e9, "LLaMA-2 70B": 70e9}
for name, params in model_sizes.items():
    mem_gb = params * 14 / 1e9  # 粗略估算（bf16 混合精度）
    print(f"{name:20s}: ~{mem_gb:.0f} GB（不含激活值）")
```

---

## 测验题

**Q1（选择）** 使用梯度累积时，`loss = loss / accumulation_steps` 这行代码的作用是：
- A. 降低学习率
- B. 防止梯度因累积而被放大 K 倍，保证与大 batch 训练等价
- C. 让 loss 更快收敛
- D. 节省显存

**答案**：B。若不除以 $K$，每次 `backward()` 的梯度相当于 $K$ 个样本的梯度，累积 $K$ 步后总梯度是 $K^2$ 倍，比真实大 batch 大 $K$ 倍。

---

**Q2（判断）** 对错判断：训练大模型时，bf16 比 fp16 更推荐，因为 bf16 的数值精度更高。

**答案**：错。bf16 的数值精度比 fp16 **低**（bf16 精度≈fp32 的低位截断），但 bf16 的**数值范围**和 fp32 相同，因此不会出现 fp16 常见的 overflow 问题。推荐 bf16 是因为稳定性更好，而不是精度更高。

---

**Q3（计算）** 训练一个 7B 参数的 LLM，使用 bf16 混合精度 + Adam 优化器，估算最少需要多少 GB 显存（不含激活值）？

**答案**：
- 模型参数（bf16）：7B × 2 bytes = 14 GB
- 梯度（fp32）：7B × 4 bytes = 28 GB
- Adam 优化器状态（fp32，m 和 v）：7B × 4 × 2 bytes = 56 GB
- 总计：≈ **98 GB**（大约 8 张 A100-80G 的显存）

---

## 课后练习（选做）
1. **对比实验**：在 GPU 上运行代码，分别用 fp32、fp16、bf16 训练，比较速度和显存（用 `torch.cuda.memory_allocated()`）
2. **阅读**：HuggingFace `TrainingArguments` 文档中 `bf16`、`gradient_accumulation_steps`、`gradient_checkpointing` 这三个参数的说明
