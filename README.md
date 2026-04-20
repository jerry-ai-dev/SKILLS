# SKILLS — 零算法基础自学大模型算法的 AI 教学仓库

> 这是笔者**个人从 0 算法基础自学大模型算法**一路沉淀下来的 Skill 仓库。
> 每一个子目录都是一个独立的 GitHub Copilot Agent Skill，配套教案、进度脚本、复习八股、阶段考试，让 AI 老师按固定流程带你从入门一路学到能做项目。

---

## 为什么有这个仓库

学大模型算法这一年多，书、视频、博客、源码笔者都试过一遍，最大的痛点是——**卡住的时候没人换个说法**。
直到开始**让大模型当老师**，才真正学进去：公式讲得准、同一个概念可以换三四种讲法、而且不嫌你烦。

但直接丢给 Copilot 一句"教我 GRPO"是不够的，它会讲得散、没主线、前后对不上。
所以笔者把自己学过一遍的经验沉淀成了这套 Skill：**把大纲定死、把讲法定死、每节课自己先学一遍再上传**，保证 AI 老师每次产出都稳定。

---

## 学习路线：4 个阶段，从 0 到能做后训练项目

笔者为自己规划了一条**从零基础到能独立完成后训练项目**的完整路线，共分 4 个阶段：

| 阶段 | 名称 | 目标 | 对应 Skill / 产出 |
|------|------|------|-------------------|
| **第一阶段** | PyTorch 入门 | 打牢 Tensor / Autograd / nn.Module / 训练流程 / Transformer / 微调基础，从 Python 跨到会用 PyTorch 写模型 | [`pytorch-teacher/`](pytorch-teacher/) |
| **第二阶段** | 后训练理论深化 | 把 RL、Policy Gradient、PPO、Reward Model、RLHF、GRPO、SFT 的原理、公式、代码全搞懂，最终读透 DeepSeek R1 论文 | [`post-training-teacher/`](post-training-teacher/) |
| **第三阶段** | 开源项目研读 | 系统阅读 TRL、Open-R1、SimpleRL-Zoo 三个项目的源码，理解工业级后训练 pipeline 是怎么拼起来的 | `code-reading-teacher` Skill（待上传） |
| **第四阶段** | 完成一个后训练项目 | 亲手跑一遍 SFT + GRPO 的完整 pipeline，把前三阶段所学真正落到训练脚本里 | 个人 repo（待上传） |

**走完 4 个阶段后，你会具备独立做一个完整后训练项目的能力**——对应到求职 JD 里那句"要求有强化学习 / RLHF 经验、熟悉 SFT 流程、理解 PPO/GRPO"，你就算是能对上号了。

---

## 当前仓库包含的 Skill

### 🟢 [`pytorch-teacher/`](pytorch-teacher/) — 第一阶段：PyTorch 入门
12 节正课 + 4 次考试，从张量一路讲到 GPT 实现与微调。面向**有 Python 基础但没接触过深度学习**的同学。详见 [pytorch-teacher/README.md](pytorch-teacher/README.md)。

### 🟢 [`post-training-teacher/`](post-training-teacher/) — 第二阶段：后训练理论深化
10 节正课 + 3 次考试 + 复习模式。RL → PPO → Reward Model → RLHF → GRPO → SFT → DeepSeek R1 主线。每节课 9 步流程 + 5 种教学风格（比喻 / 硬核 / 折中 / 工程 / 苏格拉底）可切换。详见 [post-training-teacher/README.md](post-training-teacher/README.md)。

### 🟡 `code-reading-teacher/` — 第三阶段：开源项目研读（待上传）

### 🟡 第四阶段项目（待上传）

---

## 快速开始

1. 在支持 GitHub Copilot Agent Skills 的 VS Code 环境中打开本仓库
2. 从第一阶段开始，对 Copilot 说：
   > - `学习 PyTorch` / `开始上课` → 进入第一阶段
   > - `开始后训练` / `学习 PPO` / `学习 GRPO` → 进入第二阶段
3. 老师会自动读取对应 Skill 的 `SKILL.md`、加载进度、从上次中断处继续

> 模型推荐：笔者主力使用 **Claude Opus 4.6**，讲解最有层次、类比最贴切。Sonnet 4.6 速度快可替代；GPT 系列话偏多。

---

## 目录结构

```
SKILLS/
├── README.md                 
├── pytorch-teacher/          ← 第一阶段 Skill
│   ├── SKILL.md
│   ├── README.md
│   ├── references/
│   └── scripts/
└── post-training-teacher/    ← 第二阶段 Skill
    ├── SKILL.md
    ├── README.md
    ├── references/
    └── scripts/
```

---

⭐ 如果这个仓库对你有帮助，欢迎 Star！后续第三、第四阶段也会陆续更新到这里。

---

## 关注我，获取更多资料

欢迎关注以下平台，获取更多大模型求职面试资料、面经、项目：

- 📕 **小红书**：不转到大模型不改名
- 📺 **B站**：骑猪撞宝马71
