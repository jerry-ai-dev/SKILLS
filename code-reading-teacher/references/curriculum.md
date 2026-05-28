# 阶段三：开源项目研读 — 课程大纲

共 13 课（含 3 次考试），分 3 个阶段。每个阶段：代码精读 → 关键模块分析 → 调试观察 / 参数修改 → 阶段考试。

阶段考试不考纯手写完整代码。考试重点是：读懂开源代码主干、把源码映射到训练原理、审查 AI 生成的代码 / 配置、给出最小修改建议。

**前置要求**：完成后训练理论深化（阶段二）Lesson 1-10，理解 SFT、PPO、GRPO、RLHF 的核心概念。

---

## 第一阶段：TRL 库精读 (Lesson 1-3 + Exam 1)

### Lesson 1: TRL 库全景 & SFTTrainer 源码
- TRL 库的定位：HuggingFace 的后训练一站式工具
- TRL 核心模块总览：SFTTrainer, GRPOTrainer, RewardTrainer, DPOTrainer
- SFTTrainer 源码精读：从 HuggingFace Trainer 继承了什么？加了什么？
- 数据处理：`DataCollatorForCompletionOnlyLM`（对应 Exam 1 的 completion-only label mask）
- 配置体系：`SFTConfig` 的关键参数
- **动手**：审查并运行 AI 生成的最简 SFT toy 脚本，重点观察配置、数据字段和中间输出

### Lesson 2: TRL GRPOTrainer 源码精读
- GRPOTrainer 的整体架构：训练循环怎么写的？
- 采样阶段：如何生成 G 条候选回答（`generate` 调用）
- 优势计算：组内 z-score 标准化的代码实现（对应 Exam 1 的 advantage 风险诊断）
- 策略更新：PPO-Clip loss 的代码实现（对应 Exam 1 的 GRPO 主流程定位）
- KL 惩罚：per-token KL 的代码实现（对应 Exam 1 的 KL 日志诊断）
- 参考模型管理：`ref_model` 的加载、detach、显存优化
- **动手**：阅读并注释 GRPOTrainer 的核心 50 行代码

### Lesson 3: TRL 数据流水线 & Reward 设计
- 数据格式：TRL 期望的 prompt/completion 格式
- Chat Template：如何用 `apply_chat_template` 组织多轮对话
- Reward 函数接口：TRL 的 reward function 签名和约定
- 规则奖励 vs RM 奖励的接入方式
- 训练日志与监控：WandB 集成、关键指标解读
- **动手**：审查并修改 AI 生成的数学推理 reward 函数，验证接口、答案对齐和容错逻辑

### 📝 Exam 1: TRL 库阶段考试
- 覆盖 Lesson 1-3 全部内容
- 10 题（源码定位与调用链 3 + 原理映射与运行现象解释 3 + AI 生成代码 / 配置审查 3 + 阶段四迁移清单 1），满分 100 分
- 不要求从零手写完整 SFT / GRPO 脚本或 reward 函数；重点考是否能读懂代码、发现问题、修改参数并解释原因

---

## 第二阶段：Open-R1 深度拆解 (Lesson 4-7 + Exam 2)

### Lesson 4: Open-R1 项目架构总览
- Open-R1 是什么：复现 DeepSeek R1 的开源项目
- 项目目录结构：`src/`, `configs/`, `scripts/`, `recipes/`
- 依赖关系：TRL + vLLM + DeepSpeed/FSDP
- 配置系统：YAML 配置文件怎么组织训练参数
- 与 TRL 的关系：Open-R1 在 TRL 基础上做了什么定制
- **动手**：克隆 Open-R1 仓库，画出模块依赖图

### Lesson 5: Open-R1 SFT 训练流程
- SFT 脚本入口：`run_sft.py` 逐行精读
- 数据加载：Open-R1 用什么 SFT 数据集？格式是什么？
- 模型配置：LoRA vs 全参数微调的选择
- 训练参数：学习率、batch size、warmup 的设置
- 分布式训练：DeepSpeed ZeRO Stage 的配置
- **动手**：审查 Open-R1 的 SFT 配置，改成适合 Qwen-0.5B 的 toy run 参数

### Lesson 6: Open-R1 GRPO 训练流程
- GRPO 脚本入口：`run_grpo.py` 逐行精读
- 与 TRL GRPOTrainer 的关系：Open-R1 做了哪些定制
- 采样配置：G（组大小）、temperature、max_length 的选择
- vLLM 加速推理：为什么 GRPO 需要快速推理引擎
- 训练超参数：β（KL 系数）、ε（clip 范围）、学习率的调优策略
- **动手**：理解 Open-R1 的 GRPO 配置文件，修改关键参数

### Lesson 7: Open-R1 奖励函数 & 评估体系
- Open-R1 的 reward 实现：accuracy reward + format reward
- 答案提取：正则匹配、`<answer>` 标签解析
- 评估框架：用什么 benchmark 衡量推理能力（GSM8K, MATH, AIME）
- 评估脚本：`run_eval.py` 的调用方式
- 端到端流程：从 SFT → GRPO → 评估的完整 pipeline
- **动手**：在给定 reward 函数基础上做最小修改，替换 Open-R1 默认 reward 并验证一条样例

### 📝 Exam 2: Open-R1 阶段考试
- 覆盖 Lesson 4-7 全部内容
- 10 题（架构理解 3 + 代码阅读 4 + 配置分析 3），满分 100 分
- 延续“读懂代码 + 审查配置 + 解释修改”的考试形式，不设置纯手写项目代码题

---

## 第三阶段：整合与实战规划 (Lesson 8-10 + Exam 3)

### Lesson 8: SimpleRL-Zoo 小模型 RL 实验
- SimpleRL-Zoo 是什么：专注小模型（≤3B）RL 实验的轻量框架
- 与 TRL/Open-R1 的对比：简化了什么、保留了什么
- 核心训练脚本精读：最精简的 GRPO 实现
- 资源估算：在消费级 GPU（RTX 3090/4090）上能跑什么规模
- **动手**：用 SimpleRL-Zoo 对 Qwen-0.5B 跑一个 toy GRPO 实验

### Lesson 9: 通用模式提炼 & 代码模板
- 三个项目的共同模式：对比 TRL / Open-R1 / SimpleRL-Zoo
- 训练循环模板：SFT 和 GRPO 的标准代码骨架
- 数据处理模板：prompt/completion 格式化的通用函数
- Reward 函数模板：规则奖励的标准写法
- 评估模板：生成 + 正则匹配 + 统计准确率
- **动手**：整理出一份"可复用代码清单"

### Lesson 10: 实战规划 — 你的 SFT+GRPO Pipeline
- 项目定义：Qwen-1.5B + GSM8K + SFT + GRPO
- 技术选型：用 TRL 还是 Open-R1 还是自己写？
- 资源规划：需要什么 GPU、训练多久、花多少钱
- 实验设计：基线 → SFT → GRPO 的对比实验方案
- 时间线规划：分几周完成、每周目标是什么
- **动手**：输出一份完整的项目计划书

### 🎓 Exam 3: 期末综合考试
- 覆盖 Lesson 1-10 全部内容
- 15 题（代码阅读 5 + 架构理解 4 + 实战设计 3 + 对比分析 3），满分 100 分
- 考完颁发成绩单，正式进入阶段四（实战与论文撰写）
- 期末重点考完整 pipeline 的阅读、诊断、参数审查与项目规划能力，不考闭卷手写训练框架
