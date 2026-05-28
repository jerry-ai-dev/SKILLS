# 开源项目研读教学（code-reading-teacher SKILL）

> 一个带你**系统读完 TRL、Open-R1、SimpleRL-Zoo** 三个主流后训练开源仓库的 AI 教学 Skill。
> 面向已经学完 PyTorch 基础 + 后训练理论、准备去看真实工业级代码的人。

---

## 写在最前面：为什么阶段三要单独做一份 Skill

跑完阶段二的理论课之后，笔者面对真正的开源仓库时还是懵了一下：

- 一个 `trainer.py` 几千行，从哪里开始读？
- 配置散在 YAML、`config.py`、命令行参数里，看十分钟还没找到学习率在哪里设的。
- 看完一遍代码合上电脑，脑子里只剩"哦反正它最后调了 `trainer.train()`"。

**问题不在代码难，而在缺一条主线**——理论懂了，但理论 → 源码的映射没人帮你串。
所以阶段三这套 Skill 的目标就一个：**带你把 TRL、Open-R1、SimpleRL-Zoo 三个仓库读到能口头讲清楚每个模块在干嘛**，并且能把每一段代码反向对回阶段二的公式。

和阶段一 / 阶段二不一样，本阶段：

- **不推公式、不做新理论**——理论只做映射，遇到 loss / advantage / KL 用一句话指明"这段对应阶段二哪个公式"即可；
- **不追求训练效果**——目标是"看懂、跑通、能复述"，不要 loss 下降、不刷指标；
- **不依赖专业显卡**——所有动手环节必须能在普通笔记本（CPU 或 ≤ 8GB 显存）上几分钟跑完。

### 硬约束：普通电脑可跑

这是本阶段最重要的一条约束。所有 toy 实验严格按以下策略：

1. **结构理解类**（架构 / 调用链）：只读源码 + 画结构图，不运行。
2. **数据 / Tokenize / Reward 类**：用 ~100 条小数据 + 小模型（`Qwen2.5-0.5B`、`gpt2`、`TinyLlama-1.1B`），CPU 直接跑。
3. **Trainer 类**：`max_steps=2`、`per_device_train_batch_size=1`、CPU 上 `bf16=False`，只验证"代码能走通、shape 对得上、loss 能算出"，不看收敛。
4. **GRPO 采样类**：把 vLLM 替换成 `model.generate()` 的 toy 版，`num_generations=2`、`max_new_tokens ≤ 64`。
5. **跑不动的环节**（完整 RL 收敛、大模型推理）：**只读不跑**，明确标记"留给阶段四"。

---

## 一、这个 Skill 在整个学习路线里的位置

笔者为自己规划的 4 阶段路线：

| 阶段 | 名称 | 目标 | 对应 Skill / 产出 |
|------|------|------|-------------------|
| 第一步 | **PyTorch 入门** | 打牢 Tensor / Autograd / Transformer / 微调基础 | `pytorch-teacher` Skill |
| 第二步 | **后训练理论深化** | 把 RL、PPO、GRPO、RLHF、SFT 的原理 / 公式 / 代码全部搞懂 | `post-training-teacher` Skill |
| 第三步 | **开源项目研读**（👈 当前 Skill） | 系统读 TRL、Open-R1、SimpleRL-Zoo，理解工业级 pipeline 是怎么拼起来的 | `code-reading-teacher` Skill |
| 第四步 | **完成一个后训练项目** | 亲手跑一遍 SFT + GRPO，把前三阶段所学落到训练脚本里 | 个人 repo（待进行） |

### 走完阶段三，你应该能做到

1. 读懂 TRL 中 `SFTTrainer` 和 `GRPOTrainer` 的核心实现路径；
2. 读懂 Open-R1 中 SFT / GRPO 的训练入口、配置体系、奖励函数；
3. 在 CPU 上把上述 pipeline 跑空转（`max_steps=2` 级别），并打印中间张量验证 shape；
4. 把每段代码反向映射到阶段二的理论公式（loss、advantage、KL 等）；
5. 产出一份可直接迁移到阶段四的「工程清单」（数据格式、配置项、关键函数）。

---

## 二、本 Skill 学什么：课程大纲

共 **10 节正课 + 1 节 TRL 补充课（Lesson 11: DPOTrainer） + 3 次考试**，分 3 个阶段。完整大纲见 [references/curriculum.md](references/curriculum.md)。

### 阶段一：TRL 库精读（Lesson 1–3, 11 + Exam 1）
- **Lesson 1** TRL 库全景 & `SFTTrainer` 源码（数据 collator、`SFTConfig`）
- **Lesson 2** `GRPOTrainer` 源码精读（采样 / 优势计算 / PPO-Clip / per-token KL / ref_model）
- **Lesson 3** TRL 数据流水线 & Reward 设计（chat template、reward 函数签名、规则奖励 vs RM）
- **Lesson 11** TRL `DPOTrainer` 源码精读（偏好偏对数据、`concatenated_forward`、`dpo_loss` 与 `loss_type` 变体开关）
- **Exam 1** TRL 阶段考试（源码定位 + 原理映射 + AI 生成代码审查，含 DPO 题）

### 阶段二：Open-R1 深度拆解（Lesson 4–7 + Exam 2）
- **Lesson 4** Open-R1 项目架构总览（目录结构、配置系统、与 TRL 的关系）
- **Lesson 5** Open-R1 SFT 训练流程（`run_sft.py` 入口 + 配置）
- **Lesson 6** Open-R1 GRPO 训练流程（`run_grpo.py` 入口 + vLLM 加速）
- **Lesson 7** Open-R1 奖励函数 & 评估体系（accuracy / format reward、GSM8K / MATH 评估）
- **Exam 2** Open-R1 阶段考试

### 阶段三：整合与实战规划（Lesson 8–10 + Exam 3）
- **Lesson 8** SimpleRL-Zoo 小模型 RL 实验（最精简 GRPO 实现）
- **Lesson 9** 通用模式提炼 & 代码模板（三个项目的共同骨架）
- **Lesson 10** 实战规划 — 你的 SFT+GRPO Pipeline（输出阶段四项目计划书）
- **Exam 3** 期末综合考试（颁发成绩单，正式进入阶段四）

---

## 三、本 Skill 的两套教学流程：按代码类型走

这是阶段三和阶段二最大的差别——**不再固定一种节奏**，而是根据当前要读的代码类型自动切换。

| 代码类型 | 典型例子 | 用哪套流程 |
|---|---|---|
| **库代码 / 算法代码**：含具体算法逻辑（loss、advantage、reward） | TRL 的 `SFTTrainer / GRPOTrainer`、Open-R1 的 `rewards.py` | **流程 A（详细 6 步）** |
| **项目入口脚本**：薄编排层，只做参数解析 + 调工具 + 启 Trainer | Open-R1 的 `sft.py / grpo.py`、SimpleRL-Zoo 的训练脚本 | **流程 B（精简 4 步）** |

### 流程 A — 库代码 / 算法代码（详细 6 步）

```
1. 这段代码在干嘛（来源 / 路径 / 行数 / 一句话职责）
2. 工程调用 ↔ 源码对应表（用户写的一行代码，背后跳到哪几个文件）
3. 选定贯穿样例（一条真实样本贯穿全节，追踪字段 / shape 变化）
4. 逐段精读核心代码（每段 ≤ 30 行 + 阶段二理论映射）
5. 关键问答 & 踩坑点
6. Toy 跑通 + 中间张量打印
```

### 流程 B — 项目入口脚本（精简 4 步）

```
B1. 这个脚本是干嘛的（路径 / 入口命令 / YAML 片段）
B2. 调用链一张图（Mermaid，5–8 个节点）
B3. 关键决策点速览（3–5 个"自己写时会遇到的点"）
B4. 不强制跑训练，重在让学生口头复述骨架
```

**默认教学风格**：工程优先 + 适度动手。不再做比喻 / 彩蛋 / 顺口溜，语言保持简洁、技术、中性。一次回复最多 3 个独立代码片段。

---

## 四、使用技巧

### 技巧 1：先跑结构理解类，再啃算法类

笔者的经验是——上来就硬读 `GRPOTrainer.compute_loss()` 容易劝退。
正确顺序是：

1. 先用 **流程 B** 过完所有入口脚本（Open-R1 的 `sft.py / grpo.py`），把"调用链"先在脑子里立起来；
2. 再回头用 **流程 A** 啃 TRL 里的 trainer 实现，这时你会发现"哦原来这就是 `sft.py` 最后调的那个 `trainer.train()`"。

### 技巧 2：每节课结束前的 Toy 脚本一定要在本机跑一下

不需要训练效果，但必须**亲手把脚本跑通**——这是把"读懂"变成"真的懂"的关键一步。配置已经按 CPU + 小模型 + `max_steps=2` 调好，几分钟就能看到 loss 打印出来。

### 技巧 3：阶段二的复习模式可以随时回来用

读代码时遇到"这是哪个公式"的疑惑，直接切回阶段二（`复习 GRPO` / `复习 PPO`），用 3~5 分钟把概念串一遍，再回来继续读代码。这套 Skill 体系是设计成可以来回切的，不要硬扛。

### 技巧 4：阶段三的最终产出是「工程清单」，不是代码

读完 10 课后，Lesson 10 会让你输出一份**完整项目计划书**：数据集选择、模型选型、关键配置、资源估算、时间线。**这份清单就是阶段四的入场券**，比"读懂了几个文件"重要得多。

---

## 五、快速开始

对 GitHub Copilot 说任一触发词即可开课：

> - **开始学习**：`阶段三` / `读代码` / `开源项目` / `TRL` / `Open-R1` / `code reading` / `开始阶段三`
> - **继续学习**：`继续阶段三` / `下一课`

老师会自动：
1. 读取 [SKILL.md](SKILL.md) 加载教学规范
2. 运行 `scripts/progress.py show` 查看你的进度
3. 判断当前要读的文件属于库代码还是入口脚本，自动切到流程 A 或 B

> 模型推荐：与前两阶段一致，**Claude Opus 4.6** 体验最稳定。阶段三对"长上下文 + 精确代码定位"要求更高，建议优先用 Opus。

---

## 六、目录结构

```
code-reading-teacher/
├── README.md                 ← 你正在读的这份文档
├── SKILL.md                  ← 老师的行为规范（角色、两套流程、约束）
├── progress.json             ← 学习进度（脚本自动维护）
├── scripts/
│   └── progress.py           ← 进度管理命令（show / complete / reset）
└── references/
    ├── curriculum.md         ← 完整课程大纲
    ├── prerequisites.md      ← 前置知识清单
    ├── lesson00_intro.md     ← 阶段三入门导览
    ├── lesson01_trl_sft.md   ← Lesson 1：TRL SFTTrainer
    ├── ... (lesson02 - lesson10)
    ├── exam01_trl.md         ← Exam 1：TRL 阶段考试
    ├── exam02_openr1.md      ← Exam 2：Open-R1 阶段考试
    └── exam03_final.md       ← Exam 3：期末综合考试
```

---

## 七、和阶段二的关系

| 维度 | 阶段二（post-training-teacher） | 阶段三（code-reading-teacher，本 Skill）|
|---|---|---|
| **学什么** | RL / PPO / GRPO / RLHF / SFT 原理 + 公式 | TRL / Open-R1 / SimpleRL-Zoo 源码 + 工程实现 |
| **教学节奏** | 9 步流程，5 种风格可切（比喻 / 硬核 / 折中 / 工程 / 苏格拉底）| 固定"工程优先 + 适度动手"；按代码类型切流程 A / B |
| **彩蛋 / 类比** | 有（每节课一个彩蛋小剧场）| 无（语言保持简洁、技术、中性）|
| **是否推公式** | 推 | 不推，只做"代码 ↔ 阶段二公式"的映射 |
| **最终产出** | 通过 Exam 3，理解理论闭环 | 通过 Exam 3 + 输出阶段四项目计划书 |

走完阶段三，就可以正式进入阶段四：在专业显卡上**亲手跑一遍 SFT + GRPO 的完整 pipeline**。

---

⭐ 如果这套 Skill 对你有帮助，欢迎到顶层 [SKILLS 仓库](../README.md) Star 支持。
