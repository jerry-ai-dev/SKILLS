# Exam 3: 期末综合考试

## 考试说明
- **范围**：Lesson 1-10 全部内容
- **题数**：15 题，满分 100 分
- **分布**：
  - 选择题 4 道（每题 5 分）
  - 公式推导题 4 道（每题 8 分）
  - 代码题 4 道（每题 7 分）
  - 开放设计题 3 道（每题 8 分）
- **时间建议**：90 分钟  
- **结束后**：颁发后训练理论阶段成绩单，回顾整个学习历程，规划阶段三

---

## 选择题（4 × 5分 = 20分）

### Q1【选择，5分】
以下关于 GRPO 的描述，哪一项是**错误**的？

A. GRPO 用组内相对奖励代替 GAE 估计优势，不需要 Critic 网络  
B. GRPO 的组内优势计算公式为 $\hat{A}_i = (r_i - \text{mean}) / \text{std}$  
C. GRPO 使用 PPO-Clip 目标函数（带 $\epsilon$ 的 min-clip）  
D. GRPO 因去掉 KL 惩罚项而显著节省显存  

**答案**：D  
**解析**：GRPO **仍然有 KL 惩罚项**（防止偏离参考模型），节省的是 Critic 网络的显存，不是 KL 惩罚。

---

### Q2【选择，5分】
BPE 分词器训练时，如果语料中 "ing" 出现了 10000 次，但 "un"+"happy" 只出现了 50 次，BPE 算法会：

A. 优先合并 "un"+"happy" 因为它构成一个词素  
B. 优先合并出现频率更高的符号对  
C. 同时合并两者  
D. 根据语法规则决定  

**答案**：B  
**解析**：BPE 是纯频率贪心算法，完全不考虑语言学意义，只看频率

---

### Q3【选择，5分】
混合精度训练中，以下说法正确的是：

A. bf16 精度比 fp16 更高，因此大模型推荐使用 bf16  
B. fp16 数值范围与 fp32 相同，只是精度更低  
C. bf16 数值范围和 fp32 相同，比 fp16 更不容易 overflow，推荐大模型训练使用  
D. 混合精度中，参数更新也在 fp16 下进行  

**答案**：C  
**解析**：bf16 精度低于 fp16，但范围同 fp32；fp16 范围只有 ±65504，容易 overflow；参数更新（Master weights + 梯度）仍然在 fp32

---

### Q4【选择，5分】
DeepSeek R1-Zero 的"Aha Moment"指的是：

A. 模型在某一步突然 loss 降到 0  
B. 训练到一定程度，模型自主涌现出反思和自我纠错的行为，无任何训练数据监督  
C. 研究团队发现了 GRPO 算法的数学证明  
D. 模型第一次在数学测试集上超过 GPT-4  

**答案**：B

---

## 公式推导题（4 × 8分 = 32分）

### Q5【推导，8分】
从 Policy Gradient Theorem 出发，推导 REINFORCE 的 loss 函数（带 baseline 版本）。指出 baseline 降低方差但不引入偏差的原因。

**评分标准**：
- 写出 $\nabla J = \mathbb{E}[G_t \nabla\log\pi_\theta]$（2分）
- 以负号变梯度下降 loss（1分）
- 引入 baseline：$\mathcal{L} = -\sum_t \log\pi_\theta(a_t|s_t)(G_t-b)$（2分）
- 证明 $\mathbb{E}[\nabla\log\pi \cdot b] = b\nabla\sum_a\pi = 0$（3分）

---

### Q6【推导，8分】
给定两条候选回答，奖励分别为 $r_1 = 1.0, r_2 = 0.0$（共 $G=2$ 条），求：
1. GRPO 的组内优势 $\hat{A}_1$ 和 $\hat{A}_2$
2. 若 $r_1 = r_2 = 1.0$（全部正确），优势是多少？对训练有何影响？

**答案（评分标准）**：

1. mean = 0.5, std = $\sqrt{0.5 \approx 0.707}$
   - $\hat{A}_1 = (1.0-0.5)/0.707 \approx +0.707$（2分）  
   - $\hat{A}_2 = (0.0-0.5)/0.707 \approx -0.707$（2分）

2. mean=1.0, std=0，$\hat{A}_1 = \hat{A}_2 = 0/0 = \text{NaN}$（数值问题）（2分）  
   实践处理：std+1e-8 后优势≈0，GRPO loss≈0，本组数据对参数更新几乎无贡献（2分）  
   **影响**：这是 GRPO 的局限性——如果所有 G 条回答都正确/都错误，这道题提供不了梯度信号。实践中需要设计奖励或调整 G，确保每组内有正有负

---

### Q7【推导，8分】
RLHF 训练中，有效奖励为 $r_\text{eff} = r_\phi(x,y) - \beta \cdot \text{KL}(\pi_\theta || \pi_\text{ref})$。若 $\beta = 0$，训练结果会趋向什么？若 $\beta \to \infty$，结果又会怎样？

**答案（评分标准）**：
- $\beta = 0$：无 KL 惩罚，模型会完全针对 RM 优化，极易发生 Reward Hacking（模型找到让 RM 打高分的捷径，丧失通用能力）（4分）
- $\beta \to \infty$：KL 惩罚无穷大，任何偏离参考模型的行为都会被严重惩罚，策略退化为 $\pi_\text{ref}$（即 SFT 模型），相当于没有 RL 阶段（4分）

---

### Q8【推导，8分】
梯度累积中，为什么必须对 mini-batch loss 除以 accumulation_steps K？写出不除以 K 时等效的 batch_size 和真实情况的公式对比。

**答案（评分标准）**：

**真实大 batch**：$\mathcal{L} = \frac{1}{KB} \sum_{i=1}^{KB} \ell_i$，梯度 $= \frac{1}{KB}\sum \nabla\ell_i$（2分）

**不除以K的累积**（错误做法）：
每步梯度 $= \frac{1}{B}\sum_j \nabla \ell_j$，累积K步后总梯度 $= \sum_k \frac{1}{B}\sum_j \nabla\ell_j = \frac{K}{KB}\sum \nabla\ell_i = K \times$（正确梯度）（3分）

**除以K的累积**（正确做法）：
每步 loss 为 $\ell_k/K$，梯度为 $\frac{1}{KB}\sum_j\nabla\ell_j$，K步累积后恰好等于真实大 batch（3分）

---

## 代码题（4 × 7分 = 28分）

### Q9【代码，7分】
补全 BPE 推理代码（给定一个新词和合并规则列表，输出分词结果）：

```python
def bpe_tokenize(word: str, merges: list[tuple]) -> list[str]:
    symbols = list(word) + ['</w>']  # 初始化为字符列表
    for pair in merges:
        # TODO: 将 symbols 中所有的 pair 合并
        pass
    return symbols
```

**答案**：
```python
def bpe_tokenize(word: str, merges: list[tuple]) -> list[str]:
    symbols = list(word) + ['</w>']
    for pair in merges:
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == pair:
                new_symbols.append(''.join(pair))
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols
```
**评分**：逻辑正确(4分)，边界处理正确(2分)，代码可读(1分)

---

### Q10【代码，7分】
以下代码实现了一个简单的重要性采样（Importance Sampling）校正。找出 2 个 bug：

```python
def importance_weighted_loss(log_probs_new, log_probs_old, returns):
    ratio = log_probs_new / log_probs_old   # Bug 1
    loss = ratio * returns
    return loss.mean()                       # Bug 2（符号）
```

**答案**：
- Bug 1：概率比率应在 log 空间用减法后 exp：`ratio = torch.exp(log_probs_new - log_probs_old.detach())`（3分）
- Bug 2：这是要最大化期望奖励，作为 loss 需要加负号：`return -loss.mean()`（2分）
- 还缺 detach：`log_probs_old` 需要 detach（2分）

---

### Q11【代码，7分】
以下 SFT 训练循环有一个关键的 Loss Masking 错误，找出并改正：

```python
def sft_step(model, batch):
    input_ids = batch['input_ids']           # [B, T]
    labels    = batch['input_ids'].clone()   # Bug: 没有 mask prompt
    logits    = model(input_ids).logits      # [B, T, V]
    
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    
    loss = F.cross_entropy(shift_logits, shift_labels)
    return loss
```

**答案**：`labels` 没有对 Prompt 部分做 mask。修复方法：
```python
labels = batch['input_ids'].clone()
prompt_len = batch['prompt_len']   # 每条数据的 prompt 长度
for i, plen in enumerate(prompt_len):
    labels[i, :plen] = -100   # 忽略 prompt 部分
```
（逻辑正确 5分，语法可运行 2分）

---

### Q12【代码，7分】
下面是 RLHF 中 per-token KL penalty 的计算，解释中文注释中标注的三行代码分别在做什么：

```python
def compute_kl_penalty(log_probs_policy, log_probs_ref, attention_mask):
    # Line A:
    token_kl = log_probs_policy - log_probs_ref.detach()
    # Line B:
    token_kl = token_kl * attention_mask
    # Line C:
    kl_per_sample = token_kl.sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
    return kl_per_sample
```

**答案**：
- Line A（2.5分）：计算每个 token 处新策略与参考策略的 log 概率差，即逐 token 的 KL 贡献，`.detach()` 确保不对参考模型求梯度
- Line B（2分）：过滤掉 padding 位置（mask=0），只保留有效 token 的 KL
- Line C（2.5分）：对有效 token 的 KL 求均值（不是求和），用 `clamp(min=1)` 防止全 padding 时除以零

---

## 开放设计题（3 × 8分 = 20分）

### Q13【开放，8分】
假设你要用 GRPO 训练一个 Qwen-1.5B 做 GSM8K 数学推理。请设计奖励函数：
- 奖励函数的形式（规则 or RM？为什么？）
- 具体的奖励规则（至少两条，包括格式奖励和准确率奖励）
- 如何验证答案是否正确

**参考答案（评分要点）**：
- 规则奖励（3分）：GSM8K 有标准答案，无需 RM，直接验证即可，避免 reward hacking
- 格式奖励（2分）：如答案格式是否含 `<answer>...</answer>` 标签，或是否有推理步骤
- 准确率奖励（2分）：用正则提取预测答案，与标准答案对比（数字相等则 +1）
- 验证方法（1分）：用 sympy.simplify 或直接数字比较，处理"360"和"360.0"等等价形式

---

### Q14【开放，8分】
你在做 GRPO 训练时，发现 training loss 降得很快，但在 GSM8K 测试集上准确率几乎不变甚至下降。列出 3 个可能的原因和对应的排查/解决方法。

**参考答案（每个原因+方法 2-3分）**：

1. **Reward Hacking / 奖励函数漏洞**：模型学会了"格式正确但答案随机"或特定的短语触发奖励。解决：检查奖励函数逻辑，增加更强的验证；观察模型生成的典型输出是否有规范的推理

2. **KL 惩罚太小（β 过低）**：模型过度偏离 SFT 基础模型，语言理解能力退化。解决：增大 β，或监控 KL 散度，若超过阈值就暂停训练

3. **数据分布问题**：训练数据和 GSM8K 测试数据分布不匹配（如训练用了更难或更简单的题）。解决：用与 GSM8K 类似分布的训练数据，或直接在 GSM8K 训练集上训练+验证集评估

4. **group 内全对/全错**：奖励全是 1 或全是 0，优势都接近 0，提供不了有效梯度。解决：增大 G（从 8 到 16+），提高 prompt 多样性，让每组内有正有负样本

---

### Q15【开放，8分】
对比分析：完成阶段二学习后，你对"SFT 和 GRPO 在数学推理训练中扮演的角色"有什么理解？（写出 4-5 句话，结合 DeepSeek R1 论文的发现）

**参考答案（评分要点）**：
- SFT 的角色（2分）：提供"格式规范"的能力——告诉模型应该以什么格式输出推理过程；R1-Zero 的失败说明没有 SFT 冷启动时模型的格式混乱
- GRPO/RL 的角色（3分）：在 SFT 建立格式基础后，通过**试错和奖励反馈**提升推理质量；R1 的实验表明 RL 能涌现 SFT 数据中根本不存在的反思能力
- 两者关系（2分）：SFT 是"入门"，GRPO 是"打磨"；只有 SFT 天花板低，只有 RL 初期不稳定
- 个人应用（1分）：我的项目应先确保 SFT 阶段质量（CoT 格式数据），再用 GRPO 提升推理准确率

---

## 期末成绩单模板

```
╔══════════════════════════════════════════════╗
║         后训练理论深化 — 期末成绩单            ║
╠══════════════════════════════════════════════╣
║  基础知识:  选择题      __/20分               ║
║  理论深度:  推导题      __/32分               ║
║  工程能力:  代码题      __/28分               ║
║  综合应用:  设计题      __/20分               ║
╠══════════════════════════════════════════════╣
║  总分:  __/100  等级: ______                  ║
╠══════════════════════════════════════════════╣
║  完成时间: 2026-3-XY                          ║
║  下一站:   阶段三 — 开源项目研读               ║
║  目标项目: SFT + GRPO 数学推理 (Qwen-1.5B)    ║
╚══════════════════════════════════════════════╝
```
