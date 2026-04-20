# 后训练理论深化课程大纲

共 13 课（含 3 次考试），分 3 个阶段。每个阶段：讲解课 → 代码练习 → 测验 → 阶段考试。

**前置要求**：完成 PyTorch 入门课程（阶段一）Lesson 1-12，能读懂 Transformer 代码。

---

## 第一阶段：RL 理论基础 (Lesson 1-3 + Exam 1)

### Lesson 1: 强化学习基础 & MDP
- 为什么语言模型需要强化学习？（从监督学习到 RL 的动机）
- 马尔可夫决策过程 (MDP)：状态、动作、奖励、转移概率
- 策略 (Policy)、价值函数 (Value Function)、Q 函数
- 回报 (Return) 与折扣因子 $\gamma$
- LLM 语境下的 MDP：token 即动作，句子即轨迹
- **练习**：用 Python 实现一个 mini Bandit 环境，比较贪心策略和随机策略

### Lesson 2: Policy Gradient & REINFORCE
- 策略梯度定理推导（The Policy Gradient Theorem）
- REINFORCE 算法：用 Monte Carlo 估计梯度
- 方差问题：为什么 REINFORCE 很不稳定？
- Baseline 技巧：减去均值降低方差
- **联系项目**：GRPO 的核心 loss 就是 Policy Gradient 的变体
- **练习**：用 PyTorch 实现 REINFORCE，在 CartPole 环境上跑起来

### Lesson 3: PPO 算法深入
- 从 REINFORCE 到 TRPO：为什么需要约束更新步长？
- 重要性采样 (Importance Sampling)：用旧策略的数据训练新策略
- PPO-Clip：简洁有效的近端策略优化
- 广义优势估计 (GAE-λ)：平衡偏差与方差
- Actor-Critic 架构：策略头 + 价值头
- **练习**：阅读并注释一段 mini-PPO 的关键代码，理解 clip 目标的实现

### 📝 Exam 1: RL 理论基础阶段考试
- 覆盖 Lesson 1-3 全部内容
- 10 题（选择 4 + 公式推导 3 + 代码理解 3），满分 100 分
- 考完批卷 → 评分 → 薄弱点诊断 → 错题讲解 → 针对性复习

---

## 第二阶段：RLHF 完整流程 (Lesson 4-7 + Exam 2)

### Lesson 4: Reward Model 奖励模型
- 为什么需要奖励模型？人工标注偏好数据的方式
- Bradley-Terry 偏好模型：如何从比较数据学奖励
- Reward Model 结构：在 LLM 基础上加标量输出头
- 训练 Reward Model 的 loss（偏好排序损失）
- 奖励黑客 (Reward Hacking)：过拟合奖励模型的危险
- **练习**：实现一个 mini Reward Model，用玩具偏好数据训练

### Lesson 5: RLHF 完整流程
- InstructGPT / ChatGPT 的三阶段：SFT → RM → RL
- 每个阶段的数据需求、模型变化、训练目标
- KL 散度惩罚项：防止策略偏离 SFT 模型太远
  $\mathcal{L} = \mathbb{E}[r_\theta(x,y)] - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$
- 常见陷阱：奖励模型过拟合、分布外泛化、reward hacking
- **联系项目**：你的项目是 RLHF 的简化版，用规则奖励替代 Reward Model
- **练习**：画出完整的 RLHF 数据流图（使用 renderMermaidDiagram）

### Lesson 6: GRPO 算法
- 从 PPO 到 GRPO：去掉价值网络，改用组内相对奖励
- GRPO 的 loss 公式：
  $\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\sum_t \min\left(r_t(\theta) \hat{A}_t,\, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$
- 组内奖励归一化：$\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$
- **为什么 GRPO 适合 LLM 训练**：省显存、更稳定、不需要 critic 模型
- DeepSeek R1 为什么选择 GRPO
- **练习**：实现 GRPO 的核心 loss 计算（PyTorch，约 20 行）

### Lesson 7: SFT 工程实践
- Instruction Tuning 数据格式：System / User / Assistant 三段式
- Loss Masking：只在 Assistant 回复上计算 loss，忽略 Prompt token
- Packing 技术：把多条短数据拼接成长序列，提高 GPU 利用率
- 常见 SFT 坑：数据质量、格式不一致、过拟合
- HuggingFace `Trainer` 与 `trl.SFTTrainer` 的核心参数
- **练习**：用 `trl.SFTTrainer` 微调一个 mini 模型（TinyLlama 或 GPT-2）

### 📝 Exam 2: RLHF 完整流程阶段考试
- 覆盖 Lesson 4-7 全部内容
- 10 题（选择 3 + 公式推导 3 + 代码题 4），满分 100 分

---

## 第三阶段：工程与前沿 (Lesson 8-10 + Exam 3)

### Lesson 8: 梯度累积 & 混合精度训练
- **梯度累积**：小显存模拟大 batch，`accumulation_steps` 的设置
- **混合精度（bf16/fp16）**：前向 fp16，参数更新 fp32，Loss Scaling
- bf16 vs fp16：为什么大模型更推荐 bf16（数值范围 vs 精度）
- `torch.amp.autocast` 和 `GradScaler` 的使用
- 实际训练时的显存估算公式
- **练习**：对比普通训练 vs 混合精度训练的速度和显存占用

### Lesson 9: BPE Tokenizer 原理
- 为什么不能直接用字母或词作为 token？
- Byte Pair Encoding (BPE) 算法：逐步合并高频字节对
- 词表构建过程：从字符级开始，迭代合并
- 分词实例：`"unhappiness"` → `["un", "happ", "iness"]`
- Special tokens：`<|pad|>` `<|eos|>` `<|user|>` 的作用
- HuggingFace `tokenizer.encode / decode` 底层发生了什么
- **练习**：从零实现一个玩具版 BPE 分词器（约 50 行 Python）

### Lesson 10: DeepSeek R1 论文精读
- 论文背景：为什么 R1 重要？Chain-of-Thought 推理能力从哪来？
- DeepSeek R1-Zero：**纯 RL（无 SFT cold start）** 就能涌现推理
- DeepSeek R1：SFT Cold Start → GRPO → Rejection Sampling → SFT → GRPO 完整流程
- 奖励函数设计：格式奖励 + 准确率奖励，不用 Reward Model
- **Aha Moment**：模型自主学会反思（"Wait, let me reconsider..."）
- 讨论：为什么这个方法你也能复现（简化版）？
- **练习**：用表格整理 R1 与 InstructGPT 训练流程的异同

### 🎓 Exam 3: 期末综合考试
- 覆盖 Lesson 1-10 全部内容
- 15 题（选择 4 + 公式推导 4 + 代码题 4 + 开放设计题 3），满分 100 分
- 考完颁发成绩单，总结后训练学习历程，规划阶段三（开源项目研读）
