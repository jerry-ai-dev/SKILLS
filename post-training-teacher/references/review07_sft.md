# 📖 Lesson 7 复习八股文：SFT 工程实践

> 用最精炼的概念 + 最生动的类比，帮你 5 分钟回忆本课所有核心知识点。

---

## 一、核心概念速查表

| 概念 | 一句话定义 | 生活类比 |
|------|-----------|---------|
| Instruction Tuning | 用 (指令, 高质量回答) 对做监督微调 | 给新员工发岗位手册 + 标准话术模板 |
| Chat Template | 统一的对话格式（如 ChatML），标记 system/user/assistant | 信件格式：称呼在上、正文居中、落款在下——格式不对收件人看不懂 |
| Loss Masking | 只对 assistant 回答部分计算 loss，prompt 部分标记为 -100 | 考试时只判你写的答案，不判你抄的题目 |
| Packing | 把多条短序列拼接成一条长序列，提高 GPU 利用率 | 快递打包：一个大箱子塞满 5 个小物件，不浪费空间 |
| `ignore_index=-100` | PyTorch CrossEntropyLoss 的特殊标记，遇到 -100 跳过该位置 | 阅卷老师看到 "-100" 就跳过不改——这道题不算分 |

---

## 二、关键知识卡片

### Loss Masking 原理
```
token:  [system] [user_text] [assistant_text]
label:  [ -100 ] [  -100   ] [真正的 label ]
         ↑ 跳过     ↑ 跳过     ↑ 计算 loss
```
> **人话**：Prompt 是题目，不需要模型"学会复读题目"。只让模型学回答部分。

### Packing 对比
```
不 Pack:  [seq=200][padding=1848]  ← 92% 浪费
Pack 后:  [seq1=200][seq2=500][seq3=800][seq4=500]  ← 塞满 2000
```
> **人话**：把碎片拼成整块，GPU 不做无用功。

### SFTTrainer 核心参数
```python
SFTConfig(
    max_seq_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 等效 batch=16
    learning_rate=2e-5,             # SFT 黄金学习率
    num_train_epochs=3,             # 通常 1-3
    packing=True,                   # 序列拼接
    bf16=True,                      # 混合精度
)
```

---

## 三、生动类比：培训新客服

公司要培训一批 **新客服**（LLM），让他们学会标准回答：

**SFT = 新员工培训**：
- 发一本**标准话术手册**（训练数据）
- 手册格式：`[客户问题] → [标准回答]`
- 培训时只要求员工**背标准回答**，不用背客户问题——这就是 **Loss Masking**

**Chat Template = 沟通格式规范**：
- 公司规定：先说"您好"(system)，然后复述客户需求(user)，最后给解决方案(assistant)
- 如果格式不统一，客户系统就看不懂——**必须和 tokenizer 的格式一致**

**Packing = 合并培训案例**：
- 有 100 个小问题，每个只要 30 秒讲完
- 不 pack = 每个问题占一节课 45 分钟，浪费 44 分 30 秒
- pack = 一节课塞 90 个问题，效率拉满

---

## 四、SFT 五大常见坑 & 避坑指南

| 坑 | 症状 | 解决 | 类比 |
|----|------|------|------|
| 忘记 Loss Masking | 模型复读 Prompt | label 中 prompt 部分设 -100 | 考试把题目也当答案来学 |
| 数据格式不统一 | 输出格式混乱 | 所有数据用同一 chat template | 有人写信、有人发短信——统一格式 |
| 学习率太大 (>5e-5) | 初期 loss 急降，后期遗忘 | 用 2e-5 | 练字太用力把纸戳破了 |
| 过拟合 | 训练 loss 极低但测试差 | 增加数据量或减少 epoch | 只背答案不理解 |
| Epoch 太多 | 只会说训练集里的话 | 通常 1-3 epoch | 背台词背傻了，不会自由发挥 |

---

## 五、SFT 在你的项目中的位置

```
你的项目训练流程：

Step 1: SFT（本课内容）
  → 用数学 QA 数据微调，让模型学会 <think>...</think><answer>...</answer> 格式
  → 产出：SFT 模型（也作为 GRPO 的参考模型 π_ref）

Step 2: GRPO（Lesson 6 内容）
  → 用答案正确性作为奖励，进一步提升推理能力
  → KL 惩罚参考的就是 Step 1 的 SFT 模型
```

> **SFT 的作用**：不是让模型"会做数学题"，而是让模型"知道用什么格式输出推理过程"。真正的推理能力提升靠 GRPO。

---

## 六、易混淆点 & 常见误区

1. **SFT ≠ 预训练**：预训练学"语言"（next token prediction），SFT 学"行为"（遵循指令）。
2. **Loss Masking 不是去掉 Prompt**：Prompt 仍然输入模型（模型需要看到问题才能回答），只是不对 Prompt 位置计算 loss。
3. **Packing 需要 attention mask 隔离**：拼在一起的多条序列不能互相 attend，否则会信息泄漏。
4. **`pad_token` 必须设置**：GPT-2 没有 pad token，不设置会报错。通常设为 `eos_token`。
5. **学习率 2e-5 是经验值**：不是理论最优，但对大多数 SFT 场景稳健有效。

---

## 七、记忆口诀

> **"格式统一做 SFT，Mask 掉题目学回答；Packing 塞满不浪费，学习率小慢慢来"**
> - 格式统一 = Chat Template
> - Mask 掉题目 = Loss Masking
> - Packing = 序列拼接
> - 学习率小 = 2e-5

---

## 八、自测题（快速检验）

1. Loss Masking 是如何实现的？为什么要这样做？
2. Packing 能提高 GPU 利用率的原理是什么？需要注意什么？
3. SFT 的推荐学习率和训练 epoch 数是多少？学习率太大会怎样？
4. 在你的项目中，SFT 产出的模型在 GRPO 阶段扮演什么角色？

> 如果以上问题都能不翻笔记快速回答，恭喜你——Lesson 7 已稳！
