# Exam 2: Open-R1 阶段考试

## 考试说明
- **范围**：Lesson 4-7（Open-R1 架构、SFT、GRPO、奖励与评估）
- **题数**：10 题，满分 100 分
- **分布**：架构理解 3 道（每题 10 分）+ 代码阅读 4 道（每题 10 分）+ 配置分析 3 道（每题 10 分）

---

## 架构理解题（3 × 10分 = 30分）

### Q1【架构理解，10分】
画出 Open-R1 的模块依赖关系（从底层到顶层），并说明 Open-R1 在 TRL 基础上做了哪些定制。

**答案**：transformers → trl → open-r1。Open-R1 定制了：(1) rewards.py 奖励函数，(2) YAML 配置体系，(3) 评估脚本。不修改 TRL 的训练循环。（10分）

### Q2【架构理解，10分】
Open-R1 的端到端 pipeline 分几步？每步的输入输出是什么？

**答案**：3 步。(1) SFT：输入=预训练模型+CoT数据，输出=SFT模型；(2) GRPO：输入=SFT模型+prompt+reward，输出=RL模型；(3) 评估：输入=RL模型+测试集，输出=准确率。（10分）

### Q3【架构理解，10分】
为什么 Open-R1 使用 vLLM 做推理而不是 HuggingFace generate？在什么场景下可以不用 vLLM？

**答案**：vLLM 通过 PagedAttention 和 Continuous Batching 加速推理，对 G=16 的 GRPO 采样阶段至关重要。在小模型（≤1B）或 G 很小（≤4）时可以不用，因为采样不是瓶颈。（10分）

---

## 代码阅读题（4 × 10分 = 40分）

### Q4【代码阅读，10分】
以下是 Open-R1 rewards.py 的 extract_answer 函数。为什么需要三种提取策略？如果只保留最后一种（最后一个数字），会出什么问题？

**答案**：不同训练阶段模型输出格式不同。训练初期可能没学会用标签；训练后期会用 `<answer>` 标签。只用最后一个数字的问题：推理过程中的中间数字可能被错误提取为答案。（10分）

### Q5【代码阅读，10分】
解释以下 sft.py 中这行代码的作用：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
```

**答案**：(1) `torch_dtype=torch.bfloat16` 以 bf16 精度加载模型，省一半显存；(2) `attn_implementation="flash_attention_2"` 使用 Flash Attention 2 加速注意力计算，对长序列显著降低显存和加速。（10分）

### Q6【代码阅读，10分】
以下代码是 GRPO 训练中获取参考模型 log_probs 的逻辑。解释为什么用 `torch.no_grad()` 和 `.detach()`：

```python
with torch.no_grad():
    ref_output = self.ref_model(input_ids, attention_mask=attention_mask)
    ref_log_probs = get_log_probs(ref_output.logits, input_ids)
```

**答案**：`torch.no_grad()` 不建立计算图，节省显存（参考模型不需要训练）。参考模型是固定的锚点，它的输出只用来计算 KL 惩罚，梯度不应该流过它。（10分）

### Q7【代码阅读，10分】
评估时为什么用 `do_sample=False`？如果评估时也用 `do_sample=True, temperature=0.7`，会有什么影响？

**答案**：`do_sample=False` 保证结果确定性和可复现。如果用随机采样，同一道题每次评估结果不同，无法准确比较不同模型的能力。评估结果的方差会很大。（10分）

---

## 配置分析题（3 × 10分 = 30分）

### Q8【配置分析，10分】
以下 GRPO 配置中，有效 batch size 是多少？如果 GPU 显存不够，应该优先调哪个参数？

```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
num_generations: 8
```

**答案**：有效 batch size = 4 × 4 = 16（prompt 维度），每步实际生成 16 × 8 = 128 条回答。显存不够应优先减小 per_device_train_batch_size（不影响有效 batch size 的效果，只是需要更多累积步数）。（10分）

### Q9【配置分析，10分】
对比以下两个配置，哪个训练更稳定？为什么？

```yaml
# 配置 A             # 配置 B
beta: 0.001           beta: 0.1
learning_rate: 5e-5   learning_rate: 1e-6
```

**答案**：配置 B 更稳定。配置 A 的 beta 太小（几乎没有 KL 约束）且学习率太大（RL 的 5e-5 很激进），容易导致 Reward Hacking 和策略崩溃。配置 B 的 beta 适中，学习率保守。（10分）

### Q10【配置分析，10分】
如果你要在 RTX 3090 (24GB) 上用 GRPO 训练 Qwen-1.5B，以下配置是否可行？如果不行怎么修改？

```yaml
model: Qwen/Qwen2.5-1.5B-Instruct
num_generations: 16
per_device_train_batch_size: 4
bf16: true
gradient_checkpointing: false
```

**答案**：不可行。1.5B 全参数 + G=16 + batch=4 约需 30+GB。修改：(1) 加 LoRA；(2) G 降到 8；(3) batch 降到 2；(4) 开启 gradient_checkpointing: true。（10分）
