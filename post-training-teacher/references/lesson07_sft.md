# Lesson 7: SFT 工程实践

## 教学目标
掌握 Instruction Tuning 数据格式、Loss Masking、Packing，能用 `trl.SFTTrainer` 微调一个实际模型。

---

## 讲解要点

### 1. Instruction Tuning 数据格式

经典的 ChatML 格式（GPT-4、Qwen 等广泛采用）：

```
<|im_start|>system
你是一个有用的助手。
<|im_end|>
<|im_start|>user
什么是梯度下降？
<|im_end|>
<|im_start|>assistant
梯度下降是...（模型应该生成的内容）
<|im_end|>
```

**Llama-3 / Alpaca 格式**：
```
### Instruction:
什么是梯度下降？

### Response:
梯度下降是...
```

不同模型使用不同格式，**必须跟 tokenizer 的 chat template 保持一致**，否则模型会混乱。

---

### 2. Loss Masking（关键！）

SFT 的目标：**只对 Assistant 的回答部分计算 loss**，Prompt 部分不算。

**为什么？**
- Prompt 是上下文，不是模型要学的内容
- 如果对 Prompt 计算 loss，模型会过度拟合如何"重复 prompt"，而不是学会如何回答

实现方式：将 Prompt 对应位置的标签设为 `-100`（PyTorch CrossEntropy 的 `ignore_index`）

```
token：   [system] [user_text] [assistant_text]
label：   [ -100 ] [  -100 ] [真正的label]
                    ↑ 忽略    ↑ 计算 loss
```

**一个常见错误**：忘记做 loss masking，导致模型同时拟合 system prompt 和回答，浪费计算量，效果变差。

---

### 3. Packing（序列拼接）

**问题**：大多数 SFT 数据很短（200-500 token），但 GPU 喜欢处理满长度（2048/4096）的序列。

不 pack = GPU 利用率低，大量 padding 浪费计算：
```
[seq_1 = 200 tokens][padding = 1848 tokens]  ← 92% 浪费！
```

Pack 后（把多条序列拼接，用 attention mask 隔离）：
```
[seq_1 = 200][seq_2 = 500][seq_3 = 800][seq_4 = 500] ← 塞满 2000 tokens
```

`trl.SFTTrainer` 的 `packing=True` 参数自动完成这个操作。

---

### 4. 常见 SFT 坑

| 问题 | 症状 | 解决方法 |
|------|------|---------|
| 忘记 Loss Masking | 模型倾向于复读 Prompt | 设置正确的 label=-100 |
| 数据格式不一致 | 模型输出格式混乱 | 所有数据统一用相同 chat template |
| 学习率过大 | 训练初期 loss 骤降，之后模型"遗忘"能力 | 用 2e-5 以下的小学习率 |
| 过拟合少量数据 | 训练 loss 极低但测试差 | 增加数据量或正则化 |
| 训练 epoch 过多 | 模型只会说训练集里的话 | 通常 1-3 epoch |

---

### 5. HuggingFace SFTTrainer 核心参数

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        max_seq_length=2048,     # 最大序列长度
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # 等效 batch_size=16
        learning_rate=2e-5,
        num_train_epochs=3,
        packing=True,            # 序列拼接
        bf16=True,               # 混合精度
    ),
    formatting_func=format_fn,   # 把 dataset 格式化成字符串的函数
)
```

---

## 代码示例

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===== 手动实现 Loss Masking（理解原理） =====

print("==== Loss Masking 原理演示 ====\n")

# 假设 tokenizer 把文本转成了这些 token id
# 格式：[system_tokens][user_tokens][assistant_tokens]
all_token_ids = torch.tensor([[1, 5, 8, 3, 7, 9, 4, 2, 6]])  # [1, seq_len=9]

# 假设 prompt 占前 5 个 token，response 占后 4 个
prompt_len = 5
seq_len    = all_token_ids.shape[1]

# 构造 labels：prompt 位置填 -100，response 位置填实际 token id
labels = all_token_ids.clone()
labels[0, :prompt_len] = -100   # 忽略 prompt

print(f"所有 token:  {all_token_ids[0].tolist()}")
print(f"labels:      {labels[0].tolist()}")
print(f"  (-100 = 忽略, 正数 = 计算 loss)\n")

# CrossEntropy 自动跳过 -100
vocab_size = 10
model_head = nn.Linear(16, vocab_size)  # 简化的 LM head
fake_hidden = torch.randn(1, seq_len, 16)  # 假设 LLM 输出的 hidden state

logits = model_head(fake_hidden)  # [1, seq_len, vocab_size]
# 对齐：用第 0 到 T-1 个位置的 logits 预测 1 到 T 个 labels
shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)  # [seq_len-1, vocab]
shift_labels = labels[:, 1:].reshape(-1)                   # [seq_len-1]

loss = nn.CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_labels)
print(f"Loss (只计算 response 部分): {loss.item():.4f}")

# 对比：不做 loss masking 的 loss（会更大，因为在学 prompt）
labels_no_mask = all_token_ids.clone()
shift_labels_no_mask = labels_no_mask[:, 1:].reshape(-1)
loss_no_mask = nn.CrossEntropyLoss()(shift_logits, shift_labels_no_mask)
print(f"Loss (包含 prompt 的 loss): {loss_no_mask.item():.4f}")
print("（通常 masked loss 更有效地利用梯度信号）")


# ===== 使用 trl.SFTTrainer 的示例（需要 pip install trl transformers） =====
print("\n==== trl.SFTTrainer 使用示例（代码结构）====\n")

sft_code = '''
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# 1. 准备数据（ChatML 格式）
raw_data = [
    {"prompt": "什么是梯度下降？",
     "response": "梯度下降是一种通过反复沿梯度反方向更新参数来最小化 loss 的优化算法。"},
    {"prompt": "解释反向传播",
     "response": "反向传播利用链式法则，从 loss 往回计算每个参数的梯度。"},
]

def format_fn(example):
    """把 dict 格式化成带 chat template 的字符串"""
    return f"<|user|>\\n{example[\'prompt\']}\\n<|assistant|>\\n{example[\'response\']}"

dataset = Dataset.from_list(raw_data)

# 2. 加载模型（用 GPT-2 做演示，实际用 Llama/Qwen）
model_name = "gpt2"
model     = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 没有 pad token

# 3. 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="./sft_output",
        max_seq_length=256,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        bf16=False,           # GPT-2 用 fp32，大模型改为 True
        logging_steps=1,
    ),
    formatting_func=format_fn,
)

trainer.train()
print("SFT 训练完成！")
'''
print(sft_code)
print("（实际运行时取消注释并安装 trl 库）")
```

---

## 测验题

**Q1（选择）** Loss Masking 中，将 Prompt token 对应的 label 设为 `-100` 的目的是：
- A. 让模型生成更短的回答
- B. 告诉 CrossEntropyLoss 忽略这些位置，不参与梯度计算
- C. 提高训练速度
- D. 防止模型记住 Prompt 内容

**答案**：B。PyTorch 的 `CrossEntropyLoss` 默认 `ignore_index=-100`，遇到 `-100` 标签的位置不计算 loss，梯度不会回传。

---

**Q2（填空）** SFT 训练中，通常推荐学习率约为 ___（数量级），训练 ___ epoch，学习率过大容易导致什么问题？

**答案**：学习率约 `2e-5`，训练 1-3 epoch。学习率过大容易破坏预训练权重，导致"灾难性遗忘"(catastrophic forgetting)——模型的通用语言能力退化。

---

**Q3（代码理解）** 在 SFTTrainer 的代码中，`packing=True` 会做什么？为什么它能提高 GPU 利用率？

**答案**：`packing=True` 将多条短序列拼接成一条长序列（直到达到 `max_seq_length`），用 attention mask 确保不同条数据之间不互相 attend。这样每个 batch 的 padding 极少，GPU 处理的有效 token 比例大幅提升，从而提高训练效率。

---

## 课后练习（选做）
1. **安装运行**：`pip install trl transformers datasets`，用 GPT-2 跑通上面的 SFTTrainer 示例
2. **对比实验**：同样的数据，`packing=True` vs `packing=False`，比较每步训练时间
