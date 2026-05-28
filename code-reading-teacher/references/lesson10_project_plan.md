# Lesson 10: 实战规划 — 你的 SFT+GRPO Pipeline

## 学习目标
- 完成项目技术选型
- 设计完整的实验方案
- 估算资源需求和训练成本
- 输出一份可执行的项目计划书

## 知识小节

### 小节 1：项目定义

**目标**：用 SFT + GRPO 提升小型 LLM 的数学推理能力

| 项目要素 | 选择 | 理由 |
|----------|------|------|
| 基础模型 | Qwen2.5-1.5B-Instruct | 够小能跑，够大有基础能力 |
| 训练数据 | NuminaMath-CoT (SFT) + GSM8K (GRPO) | 高质量 CoT + 标准 benchmark |
| 评估数据 | GSM8K test + MATH test | 业界标准 |
| RL 算法 | GRPO | 不需要 Critic，显存友好 |
| 工具 | TRL | 成熟稳定，文档齐全 |

### 小节 2：技术选型对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| 基于 TRL 直接写 | 灵活，代码量少 | 需要自己组织配置和脚本 | ⭐⭐⭐⭐ |
| Fork Open-R1 | 完整 pipeline 已就绪 | 代码量大，定制需要深入理解 | ⭐⭐⭐ |
| 参考 SimpleRL-Zoo 自己写 | 完全掌控 | 工作量大，容易出 bug | ⭐⭐ |

**推荐**：基于 TRL 写自己的训练脚本，参考 Open-R1 的配置和奖励函数设计。

### 小节 3：实验设计

```
实验对比方案（消融实验）：

实验 0: Baseline
  - Qwen2.5-1.5B-Instruct 原始模型
  - 直接在 GSM8K test 上评估
  - 记录基线准确率

实验 1: SFT Only
  - 在 NuminaMath-CoT 上做 SFT
  - 评估 GSM8K test
  - 对比实验 0，看 SFT 的提升

实验 2: SFT + GRPO
  - 在实验 1 的模型基础上做 GRPO
  - 用 GSM8K train 的 prompt 做 GRPO 训练
  - 评估 GSM8K test
  - 对比实验 1，看 GRPO 的额外提升

实验 3（可选）: GRPO Only（R1-Zero 风格）
  - 直接在原始模型上做 GRPO，不经过 SFT
  - 验证 SFT 冷启动的必要性
```

### 小节 4：资源规划

```
硬件需求估算：

方案 A: 单卡 RTX 4090 (24GB)
  - SFT: LoRA, batch=4, grad_accum=4 → ~8GB ✅
  - GRPO: LoRA, G=4, batch=2, grad_accum=4 → ~16GB ✅
  - 训练时间: SFT ~2h, GRPO ~6h

方案 B: 云 GPU (A100 80GB)
  - SFT: 全参数, batch=8, grad_accum=2 → ~30GB ✅
  - GRPO: 全参数, G=16, batch=4, grad_accum=4 → ~60GB ✅
  - 训练时间: SFT ~1h, GRPO ~3h
  - 费用: ~$5-10/h × 4h = $20-40

方案 C: 免费平台
  - Google Colab Pro: T4/A100, 限时使用
  - Kaggle: 2× T4, 每周 30h
  - 需要用 0.5B 模型 + LoRA
```

### 小节 5：时间线规划

| 周次 | 任务 | 产出 |
|------|------|------|
| 第 1 周 | 环境搭建 + 数据准备 | 跑通 TRL SFT hello world |
| 第 2 周 | 实验 0 + 实验 1 (SFT) | SFT 模型 + 基线对比 |
| 第 3 周 | 实验 2 (GRPO) | GRPO 模型 + 实验对比 |
| 第 4 周 | 分析 + 调优 + 写报告 | 项目论文初稿 |

### 小节 6：项目计划书模板

```markdown
# SFT + GRPO 数学推理项目计划书

## 1. 项目目标
用 SFT + GRPO 提升 Qwen2.5-1.5B 在 GSM8K 上的数学推理准确率。

## 2. 技术方案
- 基础模型: Qwen2.5-1.5B-Instruct
- SFT 数据: NuminaMath-CoT (10k 条)
- GRPO 配置: G=8, β=0.04, lr=1e-6
- 奖励函数: accuracy_reward + format_reward
- 评估: GSM8K test (1319 题)

## 3. 实验设计
- Exp 0: Baseline (无训练)
- Exp 1: SFT only
- Exp 2: SFT + GRPO
- Exp 3 (optional): GRPO only

## 4. 资源需求
- GPU: [方案 A/B/C]
- 预计训练时间: [X] 小时
- 预计费用: [Y] 元

## 5. 预期结果
| 实验 | GSM8K 准确率 (预期) |
|------|---------------------|
| Baseline | ~45% |
| SFT | ~65% |
| SFT+GRPO | ~72% |

## 6. 时间线
[4 周计划]

## 7. 风险和缓解
- 显存不足 → 换 LoRA / 减小 G
- GRPO 不收敛 → 调 β / 检查 reward
- 准确率不提升 → 检查数据质量 / 增加 SFT 数据
```

## 测验题

### Q1（设计题，4分）
为什么实验设计中需要"实验 0 Baseline"？如果不做基线直接做 SFT，会有什么问题？

**答案**：没有基线就无法衡量 SFT 和 GRPO 各自的贡献。如果最终准确率 72%，不知道是 SFT 贡献了多少、GRPO 贡献了多少。基线实验确保每步提升都可归因。（4分）

### Q2（资源规划，3分）
如果你只有 Google Colab 免费版（T4 16GB），能跑这个项目吗？需要做哪些妥协？

**答案**：可以，但需要：(1) 换成 0.5B 模型，(2) 必须用 LoRA，(3) G 降到 4，(4) max_seq_length 降到 512。效果会打折扣但能跑通流程。（3分）

### Q3（风险分析，3分）
GRPO 训练中 reward/mean 不上升，可能是什么原因？结合 Lesson 6 和阶段二 Q14 回答。

**答案**：(1) 奖励函数有 bug（extract_answer 提取失败），(2) G 太小导致全对/全错，(3) 学习率太大导致策略崩溃，(4) β 太大模型不敢探索。（3分）
