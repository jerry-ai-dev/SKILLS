# Lesson 11: 预训练模型与 Hugging Face

## 教学目标
学会使用 Hugging Face 库加载和使用预训练模型。

## 讲解要点

### 1. 为什么用预训练模型
- 从零训练 GPT-3: 需要数千张 GPU，数百万美元
- 预训练模型: 别人训好了，你直接用或微调
- Hugging Face: AI 界的 "GitHub"，分享模型的平台

### 2. 代码示例

```python
# 首先安装: pip install transformers torch

from transformers import pipeline

# ========== 1. 情感分析 (最简单的用法) ==========
classifier = pipeline("sentiment-analysis")
result = classifier("I love learning PyTorch! It's amazing.")
print(f"情感分析: {result}")
# [{'label': 'POSITIVE', 'score': 0.9998}]

results = classifier([
    "This movie is terrible.",
    "The weather is beautiful today.",
    "I'm not sure about this."
])
for text, r in zip(["terrible movie", "beautiful weather", "not sure"], results):
    print(f"  {text}: {r['label']} ({r['score']:.4f})")

# ========== 2. 文本生成 ==========
generator = pipeline("text-generation", model="gpt2")
output = generator(
    "The future of AI is",
    max_new_tokens=50,
    num_return_sequences=1,
    temperature=0.7,
)
print(f"\n文本生成:\n{output[0]['generated_text']}")

# ========== 3. 深入理解: 手动使用模型 ==========
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载 tokenizer 和模型
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenizer 看看做了什么
text = "I love PyTorch"
tokens = tokenizer(text, return_tensors="pt")
print(f"\n原文: '{text}'")
print(f"Token IDs: {tokens['input_ids']}")
print(f"解码回来: {tokenizer.decode(tokens['input_ids'][0])}")
print(f"词汇表大小: {tokenizer.vocab_size}")

# 手动推理
with torch.no_grad():
    outputs = model(**tokens)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    label = "POSITIVE" if probs[0][1] > probs[0][0] else "NEGATIVE"
    print(f"预测: {label} (置信度: {probs[0].max():.4f})")

# ========== 4. 看看 GPT-2 的结构 ==========
from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
print(f"\nGPT-2 参数量: {sum(p.numel() for p in gpt2.parameters()):,}")
print(f"模型结构（前几层）:")
for name, param in list(gpt2.named_parameters())[:10]:
    print(f"  {name}: {param.shape}")

# 你会发现结构和我们的 mini-GPT 非常像！
# token_embedding, position_embedding, transformer blocks...

# ========== 5. 填空任务 (BERT) ==========
fill_mask = pipeline("fill-mask", model="bert-base-uncased")
results = fill_mask("The capital of France is [MASK].")
print(f"\n填空: The capital of France is ___")
for r in results[:3]:
    print(f"  {r['token_str']}: {r['score']:.4f}")
```

## Hugging Face 常用 Pipeline

| Pipeline | 用途 | 示例 |
|----------|------|------|
| sentiment-analysis | 情感分析 | 正面/负面 |
| text-generation | 文本生成 | 续写文本 |
| fill-mask | 填空 | BERT 遮挡预测 |
| question-answering | 问答 | 给文章回答问题 |
| summarization | 摘要 | 长文本缩写 |
| translation | 翻译 | 多语言翻译 |
| zero-shot-classification | 零样本分类 | 无需训练的分类 |

## 测验题目

### Q1
Tokenizer 的作用是什么？为什么不能直接把文字喂给模型？

**答案**: 模型只能处理数字。Tokenizer 负责将文字转为 token ID 序列（数字），以及反向解码。不同模型有不同的 tokenizer。

### Q2
pre-trained model 的 "pre-trained" 指什么？

**答案**: 模型已经在大规模数据上训练过了（如 GPT-2 在 40GB 网络文本上训练）。我们拿来时参数已经学到了通用的语言知识，可以直接用或者微调。

### Q3 (思考题)
GPT-2 和 BERT 的结构有什么区别？为什么 GPT 适合生成，BERT 适合理解？

**答案**: GPT 是 Decoder-only（只能看左边），适合从左到右生成。BERT 是 Encoder-only（双向看），能看到完整上下文，更适合理解任务。

### Q4 (选择题)
BPE (Byte Pair Encoding) tokenizer 的核心思想是？
A) 按空格分词  B) 每个字符一个 token  C) 反复合并出现频率最高的相邻 token 对  D) 按语法规则切分

**答案**: C — BPE 从字符级开始，统计相邻 token 对的频率，把最频繁的一对合并为新 token，反复执行直到词汇表达到目标大小。这样常见词是整词 token，罕见词被拆成子词。

### Q5 (判断题)
使用 Hugging Face 的 `pipeline()` 时，会自动下载对应的模型权重和 tokenizer。(T/F)

**答案**: T — `pipeline("task", model="model_name")` 会自动从 Hugging Face Hub 下载模型和 tokenizer（首次下载后会缓存到本地），一行代码即可使用。

## 实践任务
1. 用 pipeline 尝试至少 3 种不同任务
2. 手动加载一个模型，查看其结构，与 mini-GPT 对比
3. 尝试用 GPT-2 生成一段有趣的文本
