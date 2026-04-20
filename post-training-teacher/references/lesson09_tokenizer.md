# Lesson 9: BPE Tokenizer 原理

## 教学目标
理解 BPE 分词算法的完整流程，能从零实现一个玩具版 BPE，明白 special token 的用途。

---

## 讲解要点

### 1. 为什么不直接用字母或单词作为 token？

**字母级（char-level）**：
- 优点：词表很小（26 + 符号）
- 缺点：序列太长（"hello" = 5 个 token），浪费计算

**词级（word-level）**：
- 优点：序列短
- 缺点：词表巨大（英语 100k+ 词），未知词（OOV）问题，"runs"、"running"、"ran" 是完全不同的 token

**BPE 的目标**：在字母和词之间找到平衡点——高频词/子词保持完整，低频词拆成子词组合。

---

### 2. BPE 算法：逐步合并高频字节对

**初始化**：
1. 将所有训练文本分割为字符（或字节）
2. 在每个词末尾加特殊符号（如 `</w>`）标记词边界

**迭代合并**：
重复以下步骤直到词表大小达到目标：
1. 统计所有相邻 token 对的频率
2. 找出频率最高的 token 对 $(a, b)$
3. 将所有 $(a, b)$ 合并为新 token $ab$
4. 将合并规则记录到合并列表

**示例**：
```
初始词表: [h, e, l, l, o, </w>]
文本: "hello hello hello world"

第1步: "l l" 出现 3 次最多 → 合并为 "ll"
第2步: "h e" 出现 3 次 → 合并为 "he"
第3步: "he ll" 出现 3 次 → 合并为 "hell"
...以此类推
```

---

### 3. BPE 用于分词（Inference）

训练好 BPE 后，对新词的分词过程：
1. 将词拆为字符序列
2. 按**训练时记录的合并顺序**（优先级）逐步合并
3. 直到无法继续合并

例如，训练时学到了规则 `("un", "happ") → "unhapp"` 和 `("unhapp", "iness") → "unhappiness"`：
- `"unhappiness"` → `u n h a p p i n e s s` → `un h a pp i ness` → `un happ i ness` → `unhapp iness` → `unhappiness`

---

### 4. Byte-level BPE（现代 LLM 使用的版本）

原始 BPE 用字符作为基础单元，无法处理 Unicode 字符。
现代方法（GPT-2、Llama 等）用**字节（byte）**作为基础单元：
- 词表初始化为 256 个字节（0-255）
- 可以处理任何语言，无 OOV 问题
- `tiktoken`（OpenAI）和 HuggingFace `tokenizer` 库都支持

---

### 5. Special Tokens

| Token | 用途 |
|-------|------|
| `<\|pad\|>` / `<pad>` | 补齐 batch 中不同长度的序列，loss 中忽略 |
| `<\|eos\|>` / `</s>` | 序列结束标记，生成时看到这个停止 |
| `<\|bos\|>` / `<s>` | 序列开始标记 |
| `<\|user\|>` / `<\|im_start\|>` | Chat 格式中标记对话轮次 |
| `<\|think\|>` | DeepSeek R1 的自定义思维链 token |

**重要**：在 SFT 时，`<pad>` 对应的 label 设为 `-100`（Loss Masking 的一部分）。

---

## 代码示例

```python
from collections import Counter, defaultdict
import re

# ===== 从零实现玩具版 BPE =====
print("==== 玩具版 BPE 实现 ====\n")

def get_vocab(corpus: list[str]) -> dict:
    """
    初始化词表：每个词拆为字符，词尾加 </w> 标记
    返回: {词表示: 出现次数}
    """
    vocab = Counter()
    for word in corpus:
        # 用空格分隔字符，词尾加 </w>
        chars = ' '.join(list(word)) + ' </w>'
        vocab[chars] += 1
    return dict(vocab)


def get_pairs(vocab: dict) -> Counter:
    """统计所有相邻符号对的频率"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs


def merge_pair(pair: tuple, vocab: dict) -> dict:
    """将 vocab 中所有的 pair 合并为新符号"""
    new_vocab = {}
    bigram = ' '.join(pair)        # "l l"
    replacement = ''.join(pair)    # "ll"
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab


def bpe_train(corpus: list[str], num_merges: int = 15):
    """
    训练 BPE 分词器
    corpus:     文本语料（单词列表）
    num_merges: 合并次数（= 最终词表大小 - 初始词表大小）
    """
    vocab = get_vocab(corpus)
    merges = []  # 记录所有合并规则（按优先级排序）

    print(f"初始词表（前 5 条）：")
    for w, c in list(vocab.items())[:5]:
        print(f"  '{w}' : {c}")
    print()

    for i in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)
        print(f"  合并 {i+1:2d}: {best_pair[0]!r:8s} + {best_pair[1]!r:8s} "
              f"→ {''.join(best_pair)!r:12s}  (频率={pairs[best_pair]})")

    return vocab, merges


# 语料
corpus = ["low", "lowest", "newer", "new", "wider", "new", "new", "low", "low",
          "lower", "lower", "lower", "lower", "newer", "newer", "wider"]

print(f"语料词频：{Counter(corpus)}\n")
vocab, merges = bpe_train(corpus, num_merges=10)

print(f"\n最终词表（学到的子词）：")
final_tokens = set()
for word in vocab:
    final_tokens.update(word.split())
print(sorted(final_tokens))

# ===== HuggingFace Tokenizer 演示 =====
print("\n==== HuggingFace Tokenizer 使用演示 ====\n")

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    texts = ["Hello, world!", "unhappiness", "PyTorch is awesome"]
    for text in texts:
        token_ids = tokenizer.encode(text)
        tokens    = tokenizer.convert_ids_to_tokens(token_ids)
        decoded   = tokenizer.decode(token_ids)
        print(f"原文:   {text!r}")
        print(f"Token:  {tokens}")
        print(f"ID:     {token_ids}")
        print(f"还原:   {decoded!r}\n")

except ImportError:
    print("（未安装 transformers，跳过演示）")
    print("示例输出：")
    print("  'Hello, world!' → ['Hello', ',', 'Ġworld', '!']  （Ġ表示空格前缀）")
    print("  'unhappiness'   → ['un', 'h', 'app', 'iness']")
```

---

## 测验题

**Q1（选择）** BPE 算法每次合并的依据是：
- A. 随机选一对符号合并
- B. 选择语言学上最有意义的词对
- C. 选择在语料中出现**频率最高**的相邻符号对
- D. 选择长度最短的符号对

**答案**：C。BPE 是纯频率驱动的贪心算法，完全不依赖人工语言学知识。

---

**Q2（概念）** 为什么现代 LLM（如 GPT-2、Llama）使用 Byte-level BPE，而不是 Character-level BPE？

**答案**：Byte-level BPE 以字节（0-255）为基础单元，天然支持所有 Unicode 字符和任何语言，不存在"未知字符"问题。Character-level BPE 对非英文字符（中文、emoji 等）处理较差，需要额外处理 Unicode 边界问题。

---

**Q3（实操）** 用 HuggingFace tokenizer，以下哪种 token 化结果是正确的？
- A. `"running"` → `["run", "##ning"]`
- B. `"running"` → `["Ġrun", "ning"]`
- C. `"running"` → `["running"]`（单个完整 token）
- D. 以上都可能，取决于具体模型的词表

**答案**：D。不同模型词表不同，"running" 在 GPT-2 的 BPE 词表里是单个 token（因为频率高），而在其他模型里可能被拆分。`##` 前缀是 BERT 的 WordPiece 格式，`Ġ` 是 GPT-2 的 BPE 空格前缀。

---

## 课后练习（选做）
1. **扩展代码**：实现 `bpe_tokenize(word, merges)` 函数，给定一个新词和记录的合并规则，输出它的 BPE 分词结果
2. **实验**：用 `gpt2` tokenizer，测试几个中文、日文句子，观察字节 token 的形态（中文字会被拆成多个字节 token）
