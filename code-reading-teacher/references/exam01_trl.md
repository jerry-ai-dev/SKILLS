# Exam 1: TRL 库阶段考试

## 考试定位

本考试覆盖 Lesson 1-3（`SFTTrainer`、`GRPOTrainer`、数据流水线与 Reward 设计）。

考试目标不是考学生能不能从零手写训练代码，而是考 4 件事：

1. **能不能读懂源码主干**：看到公开 API，知道会进入哪些类 / 函数 / 文件；
2. **能不能把源码映射回原理**：知道 label mask、advantage、KL、clip、reward 分别对应训练中的哪一步；
3. **能不能审查 AI 生成的代码 / 配置**：发现参数、接口、数据格式、reward 逻辑中的问题；
4. **能不能做最小工程修改**：不是从零写代码，而是在已有代码 / AI 生成代码上改参数、改一两行逻辑，并说明影响。

## 考试形式

- **题数**：10 题，满分 100 分；
- **题型分布**：
  - 源码定位与调用链：3 题 × 10 分；
  - 原理映射与运行现象解释：3 题 × 10 分；
  - AI 生成代码 / 配置审查：3 题 × 10 分；
  - 阶段四迁移清单：1 题 × 10 分；
- **允许**：查源码、查课程笔记、使用 AI 生成候选代码；
- **不考**：纯手写完整训练脚本、纯手写完整 reward 函数、背诵 API 参数；
- **提交形式**：每题提交 3 部分：
  1. 定位：文件 / 类 / 函数 / 关键代码段；
  2. 解释：这段代码在训练流程里负责什么；
  3. 修改建议：如果要用于阶段四项目，哪些参数或逻辑需要调整。

---

## A. 源码定位与调用链（3 × 10 分 = 30 分）

### Q1【SFTTrainer 调用链定位，10 分】

给定一段用户侧代码：

```python
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

要求说明：

1. 这段公开 API 背后会进入 TRL 的哪个文件、哪个核心类；
2. `SFTTrainer` 相比 HuggingFace `Trainer` 主要额外处理了哪些事情；
3. 训练时真正算 token-level cross entropy loss 的位置，主要来自 TRL 自己，还是继承自 Transformers Trainer。

**评分要点**：

- 能定位到 `trl/trainer/sft_trainer.py` 与 `SFTTrainer`；
- 能说明它主要负责数据格式化、tokenize / collator / packing / peft 等训练前工程处理；
- 能说明底层训练循环和常规 LM loss 很多来自 Transformers `Trainer` / causal LM 模型本身，而不是 TRL 从零重写。

### Q2【Completion-only label mask 读懂，10 分】

给定一条样本：

```text
USER: 你好
ASSISTANT: 答案是 8
```

要求说明：

1. 如果使用 completion-only 训练，哪些 token 的 `labels` 应该被置为 `-100`；
2. 为什么 `ASSISTANT:` 及其之前的 prompt 部分通常不参与 loss；
3. 如果 response template 没匹配上，训练会出现什么风险。

**评分要点**：

- 能说清楚 prompt / response 的 loss mask 边界；
- 能解释 `-100` 是 PyTorch cross entropy 的 ignore index；
- 能指出 template 不匹配会导致 mask 错位，轻则训练目标错误，重则整条样本 loss 异常。

### Q3【GRPOTrainer 主流程定位，10 分】

要求按顺序写出 `GRPOTrainer` 一次训练 step 的主干流程：

1. prompt batch 输入；
2. 每个 prompt 生成 G 条 completion；
3. 调用 reward functions / reward model 得分；
4. 组内归一化得到 advantage；
5. 计算 policy ratio / clip loss / KL 惩罚；
6. 反向传播更新 policy model。

**评分要点**：

- 顺序正确；
- 能区分 policy model、reference model、reward function 的角色；
- 能指出 G 条 completion 的目的不是扩大 batch，而是做组内相对比较。

---

## B. 原理映射与运行现象解释（3 × 10 分 = 30 分）

### Q4【SFT loss mask 原理映射，10 分】

给定一个 batch：

```python
input_ids = [USER, 你, 好, ASST, 答, 案, 是, 8]
labels    = [-100, -100, -100, -100, 答, 案, 是, 8]
```

要求解释：

1. 这个 batch 的训练目标是什么；
2. 哪些位置会产生梯度；
3. 这个设计对应阶段二 SFT 的哪个核心目标。

**评分要点**：

- 能说明只训练 assistant response；
- 能说明 prompt token 仍作为上下文输入，但不作为预测目标；
- 能映射到“最大化示范答案 token 的条件似然”。

### Q5【GRPO advantage 除零风险，10 分】

给定 AI 生成的解释：

```python
advantages = (rewards - rewards.mean(dim=-1, keepdim=True)) / rewards.std(dim=-1, keepdim=True)
```

AI 说：“这行没有问题，因为标准化就是减均值除标准差。”

要求审查这句话：

1. 这行代码在 reward 全相等时会发生什么；
2. 应该增加什么保护；
3. 这类问题在训练日志里可能表现成什么现象。

**评分要点**：

- 能指出 `std=0` 导致 NaN；
- 能提出 `+ epsilon` 或在标准差过小时置零 advantage；
- 能联系到 loss / grad_norm / reward 指标出现 NaN 或训练中断。

### Q6【KL 上升但 reward 上升的诊断，10 分】

给定训练日志：

```text
reward/mean: 0.72 -> 0.81 -> 0.88
kl:          2.1  -> 7.8  -> 16.4
loss:       -0.3  -> -0.9  -> -1.7
```

要求判断：

1. 这是不是单纯的好现象；
2. 最可能的风险是什么；
3. 从参数角度给出 3 个优先修改建议。

**评分要点**：

- 能指出 reward 上升但 KL 过高可能是策略偏离参考模型过远；
- 能提到 reward hacking / 输出风格崩坏 / 过拟合奖励；
- 能建议增大 `beta`、降低 learning rate、降低 temperature 或缩小训练步数 / clip 范围。

---

## C. AI 生成代码 / 配置审查（3 × 10 分 = 30 分）

### Q7【审查 AI 生成的 SFT 配置，10 分】

AI 给出如下配置片段：

```python
SFTConfig(
    output_dir="outputs/sft",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    max_steps=1000,
    bf16=True,
    packing=True,
)
```

假设你在普通笔记本或 8GB 显存机器上只想跑通 toy SFT 流程。要求：

1. 标出至少 4 个需要审查或修改的参数；
2. 给出更适合 toy run 的参数建议；
3. 说明每个修改影响显存、速度、稳定性还是数据格式。

**评分要点**：

- 能审查 batch size、max_steps、bf16、packing、learning rate、gradient accumulation 等；
- 能给出 `max_steps=1~2`、小 batch、关闭或谨慎开启 `bf16`、必要时关闭 `packing` 等建议；
- 不是机械改小，而是能说明修改原因。

### Q8【审查 AI 生成的 reward 函数，10 分】

AI 给出如下 reward 函数：

```python
def accuracy_reward(completions, answer=None, **kwargs):
    rewards = []
    for completion in completions:
        predicted = extract_answer(completion)
        if predicted == answer:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards
```

要求：

1. 指出这个函数在 TRL reward 接口下的至少 2 个问题；
2. 不需要从零重写，只说明最小修改方向；
3. 说明为什么这些问题会影响 GRPO 的 advantage 质量。

**评分要点**：

- 能指出 `answer` 通常是一批答案，应与 `completions` 对齐；
- 能指出字符串精确相等过于脆弱，应做 normalize / 数值比较 / 容错；
- 能说明 reward 错误会直接污染组内排序和 advantage。

### Q9【审查 GRPO 采样与 KL 参数，10 分】

AI 给出如下 GRPO 参数建议：

```python
num_generations = 1
temperature = 0.0
beta = 0.0
max_completion_length = 1024
```

要求：

1. 逐项判断这些参数对 GRPO 是否合理；
2. 给出适合 toy run 的替代建议；
3. 说明 `num_generations`、`temperature`、`beta` 三者分别影响什么。

**评分要点**：

- 能指出 `num_generations=1` 失去组内比较意义；
- 能指出 `temperature=0` 会降低候选多样性；
- 能指出 `beta=0` 取消 KL 约束，训练风险增大；
- 能建议 `num_generations=2`、较小 `max_completion_length`、非零 `beta`、适度 temperature。

---

## D. 阶段四迁移清单（1 × 10 分 = 10 分）

### Q10【把 TRL 阅读结果迁移到自己的项目，10 分】

假设阶段四要做一个 `Qwen-1.5B + 数学数据 + SFT + GRPO` 项目。要求整理一份最小工程清单，包含：

1. 数据字段：SFT 与 GRPO 分别至少需要哪些字段；
2. 配置项：每个阶段最优先确认的 5 个参数；
3. 调试点：每个阶段至少 2 个必须打印 / 断点观察的中间对象；
4. 风险项：至少 3 个最可能踩坑的问题。

**评分要点**：

- 能把 Lesson 1-3 的源码阅读转化成阶段四 checklist；
- 能覆盖数据、tokenizer / chat template、trainer config、reward、日志指标；
- 重点是可执行的工程检查项，不是泛泛总结。

---

## 考试通过标准

- **通过**：≥ 70 分，且 Q1-Q3 中至少 2 题通过，Q7-Q9 中至少 2 题通过；
- **优秀**：≥ 90 分，能够主动指出源码 / 配置和阶段二理论之间的对应关系；
- **未通过后的复习路径**：
  - Q1/Q2/Q4 弱：回看 Lesson 1 与 Lesson 3 的数据处理 / label mask；
  - Q3/Q5/Q6/Q9 弱：回看 Lesson 2 的 GRPO 主流程、advantage、KL、采样参数；
  - Q7/Q8/Q10 弱：回看 Lesson 1-3 的工程参数表、reward 接口、阶段四 checklist。
