# Lesson 4: Reward Model 奖励模型

## 教学目标
理解 Reward Model 的训练方式（Bradley-Terry 模型），能搭建并训练一个 mini Reward Model。

---

## 讲解要点

### 1. 为什么需要 Reward Model？

人类对 LLM 回答质量的偏好很难用简单规则描述。我们能做到的是：
- 给两个回答 $y_w$（胜者）和 $y_l$（败者），问"哪个更好？"
- 收集大量这样的偏好对 $(x, y_w, y_l)$

**Reward Model（RM）** 的目标：学到一个函数 $r_\phi(x, y)$，能对任意 (提示, 回答) 打分，分数反映人类偏好。

---

### 2. Bradley-Terry 偏好模型

**模型假设**：对于两个回答 $y_w$ 和 $y_l$，人类选择 $y_w$ 的概率为：

$$P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

其中 $\sigma$ 是 sigmoid 函数。

**训练 loss**（最大化正确偏好的对数似然，等价于最小化下式）：

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log\sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

**直觉**：
- 如果 $r(y_w) \gg r(y_l)$：$\sigma(\text{large}) \to 1$，$\log \to 0$，loss 小 ✅
- 如果 $r(y_w) \approx r(y_l)$：$\sigma(0) = 0.5$，$\log \to -0.69$，loss 大 ❌
- 训练目标：让 $r(y_w) - r(y_l)$ 尽量大

---

### 3. Reward Model 结构

在实践中，Reward Model 通常基于一个 LLM backbone + 标量输出头：

```
输入: [Prompt x + Response y]  →  LLM backbone  →  最后一个 token 的隐状态  →  Linear(d, 1)  →  标量分数 r
```

- Backbone 通常从 SFT 模型初始化（已经理解语言）
- 只有线性头 + 部分 backbone 层需要微调

---

### 4. Reward Hacking（奖励黑客）

**问题**：RL 训练时，模型会找到让 RM 打高分的"捷径"，而不是真正提高质量：
- 生成异常长的回答（某些 RM 偏好更长）
- 重复某些讨好 RM 的短语
- 回答方式分布与 RM 训练数据差异扩大后，RM 预测失准

**防止方法**：
1. KL 散度惩罚：保持策略不偏离 SFT 模型太远
2. 定期更新 RM（Iterative RLHF）
3. 用多个 RM 取最小分（保守奖励）

---

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== Mini Reward Model =====
class MiniRewardModel(nn.Module):
    """
    玩具版 Reward Model：输入是一个"质量特征向量"（模拟句子嵌入）
    真实版本中 backbone 会是 GPT/BERT，这里用简单 MLP 代替
    """
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # 标量输出头：输出一个分数
        self.reward_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, input_dim] → reward: [batch]"""
        features = self.backbone(x)
        reward = self.reward_head(features).squeeze(-1)  # [batch]
        return reward


# ===== Bradley-Terry 偏好 Loss =====
def preference_loss(
    reward_chosen:  torch.Tensor,   # 胜者的奖励 [batch]
    reward_rejected: torch.Tensor,  # 败者的奖励 [batch]
) -> torch.Tensor:
    """
    Loss = -E[ log σ(r_chosen - r_rejected) ]
    """
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    return loss


# ===== 数据集 =====
class PreferenceDataset(Dataset):
    """
    偏好数据集：每条数据是 (chosen_features, rejected_features)
    真实场景中，chosen/rejected 是两个回答的 token embedding，
    这里用随机向量 + 简单规则模拟"好"和"差"的回答
    """
    def __init__(self, n_samples: int = 200, input_dim: int = 32):
        torch.manual_seed(0)
        # "好"回答：特征均值更高
        self.chosen   = torch.randn(n_samples, input_dim) + 0.5
        # "差"回答：特征均值更低
        self.rejected = torch.randn(n_samples, input_dim) - 0.5

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, idx):
        return self.chosen[idx], self.rejected[idx]


# ===== 训练 Reward Model =====
input_dim = 32
model     = MiniRewardModel(input_dim=input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataset   = PreferenceDataset(n_samples=200, input_dim=input_dim)
loader    = DataLoader(dataset, batch_size=16, shuffle=True)

print("==== 训练 Reward Model ====\n")
for epoch in range(5):
    total_loss = 0
    correct = 0
    for chosen, rejected in loader:
        r_chosen   = model(chosen)
        r_rejected = model(rejected)

        loss = preference_loss(r_chosen, r_rejected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # 准确率：chosen 分数 > rejected 分数 的比例
        correct += (r_chosen > r_rejected).sum().item()

    n = len(dataset)
    acc = correct / n
    print(f"Epoch {epoch+1}/5  Loss={total_loss/len(loader):.4f}  Accuracy={acc:.1%}")

# ===== 验证 =====
model.eval()
with torch.no_grad():
    test_good = torch.randn(1, input_dim) + 0.5
    test_bad  = torch.randn(1, input_dim) - 0.5
    r_good = model(test_good).item()
    r_bad  = model(test_bad).item()
    print(f"\n好回答的奖励分数:  {r_good:.3f}")
    print(f"差回答的奖励分数:  {r_bad:.3f}")
    print(f"差值 (应为正数):   {r_good - r_bad:.3f}")
```

---

## 测验题

**Q1（选择）** Bradley-Terry 模型中，训练 Loss $\mathcal{L} = -\log\sigma(r_w - r_l)$ 越小，说明：
- A. 模型认为两个回答一样好
- B. 模型正确地给胜者更高分（分差越大 loss 越小）
- C. 模型的参数更少
- D. 奖励分数更接近 0

**答案**：B。$\sigma(r_w - r_l)$ 越大代表预测越正确，取 $-\log$ 后越小。

---

**Q2（简答）** 什么是 Reward Hacking？为什么 KL 惩罚项可以缓解这个问题？

**答案要点**：Reward Hacking 是模型找到让 RM 打高分的"捷径"而非真正提升质量的现象。KL 惩罚项 $\beta \cdot \text{KL}(\pi_\theta || \pi_\text{ref})$ 让策略不能偏离 SFT 参考模型太远，相当于限制模型只能在 SFT 已知行为空间的邻域内优化，而 RM 在这个范围内的预测相对准确。

---

**Q3（代码）** 在 `preference_loss` 函数中，如果把 `r_chosen - r_rejected` 改成 `r_rejected - r_chosen`，训练会发生什么？

**答案**：模型会被训练成给差回答打更高分、给好回答打更低分——即学反了！这说明偏好数据的 chosen/rejected 标记的方向非常重要，标注错误会导致 RM 完全学错。

---

## 课后练习（选做）
1. **修改代码**：在 `preference_loss` 中增加一个 margin，让 loss 变为 $-\log\sigma(r_w - r_l - m)$，$m=0.5$。观察训练有何变化？
2. **思考**：为什么真实的 RM 通常以 LLM 为 backbone 而不是用 BERT 这样的小模型？
