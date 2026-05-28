# Lesson 4: Open-R1 项目架构总览

## 学习目标
- 理解 Open-R1 项目的目标和定位
- 掌握项目的目录结构和模块关系
- 理解 Open-R1 与 TRL 的关系
- 能读懂 YAML 配置文件

## 知识小节

### 小节 1：Open-R1 是什么

Open-R1 是由 HuggingFace 发起的开源项目，目标是复现 DeepSeek R1 的训练流程。

- **GitHub**: https://github.com/huggingface/open-r1
- **核心功能**：基于 TRL 实现完整的 SFT → GRPO 训练 pipeline
- **与 DeepSeek R1 的关系**：Open-R1 参考了 R1 论文的方法论，但使用开源模型和数据

### 小节 2：项目目录结构

```
open-r1/
├── src/open_r1/
│   ├── grpo.py          # GRPO 训练入口脚本
│   ├── sft.py           # SFT 训练入口脚本
│   ├── evaluate.py      # 评估脚本
│   ├── rewards.py       # 奖励函数定义
│   ├── configs.py       # 自定义配置类
│   └── utils.py         # 工具函数
├── recipes/
│   ├── Qwen2.5-1.5B-Instruct/
│   │   ├── sft/config.yaml      # SFT 配置
│   │   └── grpo/config.yaml     # GRPO 配置
│   └── DeepSeek-R1-Distill-Qwen-1.5B/
│       └── ...
├── Makefile             # 常用命令快捷方式
├── setup.py
└── requirements.txt
```

### 小节 3：核心模块关系

Open-R1 的模块依赖关系（从底层到顶层）：

```
transformers (模型层)
    ↓
trl (训练框架层：SFTTrainer, GRPOTrainer)
    ↓
open-r1/src/ (项目定制层：奖励函数、配置、脚本)
    ↓
open-r1/recipes/ (实验配置层：YAML 文件)
```

Open-R1 并不重写 TRL 的训练器，而是：
1. **定制奖励函数**（rewards.py）
2. **组织配置文件**（recipes/）
3. **提供训练入口**（sft.py, grpo.py）

### 小节 4：YAML 配置文件解读

```yaml
# recipes/Qwen2.5-1.5B-Instruct/grpo/config.yaml（简化版）

# 模型配置
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
torch_dtype: bfloat16

# GRPO 专用参数
num_generations: 8          # G = 8
temperature: 0.7
beta: 0.04                  # KL 惩罚系数 β
max_completion_length: 1024

# 训练参数
per_device_train_batch_size: 4
gradient_accumulation_steps: 4   # 有效 batch = 4 × 4 = 16
learning_rate: 1.0e-6
num_train_epochs: 1
bf16: true
gradient_checkpointing: true     # 用时间换显存

# 数据配置
dataset_name: AI-MO/NuminaMath-TIR
dataset_config: default

# 保存与日志
output_dir: ./output/grpo
logging_steps: 1
save_strategy: "steps"
save_steps: 100
```

### 小节 5：Open-R1 与 TRL 的关系

| 组件 | TRL 提供 | Open-R1 定制 |
|------|----------|-------------|
| SFTTrainer | ✅ 完整实现 | 只写配置，不改代码 |
| GRPOTrainer | ✅ 完整实现 | 只写配置，不改代码 |
| 奖励函数 | ❌ 需要用户实现 | ✅ rewards.py 提供多种奖励 |
| 配置管理 | 基础 TrainingArguments | ✅ YAML 配置 + recipes 体系 |
| 评估 | ❌ 不提供 | ✅ evaluate.py + lighteval |

## 测验题

### Q1（架构理解，4分）
Open-R1 的 grpo.py 里是否重写了 GRPOTrainer 的训练循环？如果没有，它主要做了什么？

**答案**：没有重写训练循环。grpo.py 主要做的是：(1) 解析 YAML 配置，(2) 加载模型和数据集，(3) 注册自定义奖励函数，(4) 调用 TRL 的 GRPOTrainer.train()。训练核心逻辑完全由 TRL 提供。（4分）

### Q2（配置分析，3分）
上面的 YAML 配置中，有效 batch size 是多少？一共训练多少个 epoch？

**答案**：有效 batch size = 4 × 4 = 16，训练 1 个 epoch。（3分）

### Q3（概念理解，3分）
`gradient_checkpointing: true` 是什么意思？为什么 Open-R1 要开启它？

**答案**：梯度检查点是一种用时间换显存的技术——前向传播时不保存所有中间激活值，反向传播时重新计算。GRPO 需要同时加载策略模型和参考模型，显存紧张，开启 gradient_checkpointing 可以降低显存占用。（3分）
