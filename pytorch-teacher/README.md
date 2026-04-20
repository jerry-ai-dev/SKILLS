# PyTorch Teacher — 入门深度学习 / PyTorch 的导师

> 一位耐心的 AI 导师，带你从零开始系统学习 PyTorch 与深度学习。

## 简介

**PyTorch Teacher** 是一个专为深度学习初学者设计的 AI 教学技能（Skill）。  
它以循序渐进的课程体系，结合讲解、代码练习、测验与总结，帮助你从零基础到掌握 Transformer、GPT 等前沿模型。

## 课程体系

| 课程 | 主题 |
|------|------|
| Lesson 01 | 张量基础（Tensor） |
| Lesson 02 | 自动微分（Autograd） |
| Lesson 03 | 神经网络模块（nn.Module） |
| Lesson 04 | 模型训练流程 |
| Lesson 05 | 数据加载与处理 |
| Lesson 06 | 卷积神经网络（CNN） |
| Lesson 07 | 序列模型（RNN/LSTM） |
| Lesson 08 | 注意力机制（Attention） |
| Lesson 09 | Transformer 架构 |
| Lesson 10 | GPT 模型实现 |
| Lesson 11 | 预训练模型使用 |
| Lesson 12 | 模型微调（Fine-tuning） |

## 特性

- 每节课包含：讲解 → 代码练习 → 测验 → 总结
- 自动跟踪学习进度，支持随时继续或复习
- 适合有 Python 基础但没有深度学习经验的学习者
- 涵盖从基础张量运算到 GPT 的完整学习路径

## 使用方式

在支持 GitHub Copilot Agent Skills 的环境中，说：

- `学习 PyTorch` / `开始上课`
- `继续学习` / `下一课`
- `复习第X课`
- `pytorch lesson`

导师会自动根据你的进度继续教学。

## 适合人群

- 有 Python 基础，想入门深度学习的开发者
- 想系统学习 PyTorch 框架的学生
- 希望了解 Transformer / GPT 原理的工程师

## 目录结构

本 Skill 位于仓库的 `pytorch-teacher/` 子目录下（仓库根目录可并列放置其他 Skill）：

```
SKILLS/
└── pytorch-teacher/          # 当前 Skill
    ├── SKILL.md              # Skill 配置与触发规则
    ├── README.md             # 本文件
    ├── progress.json         # 学习进度记录
    ├── references/           # 课程参考资料
    │   ├── curriculum.md     # 课程大纲
    │   ├── lesson01_tensor.md
    │   ├── ...
    │   └── exam04_final.md   # 期末考试
    └── scripts/
        └── progress.py       # 进度管理脚本
```

---

⭐ 如果这个项目对你有帮助，欢迎 Star！

---

## 关注我，获取更多资料

欢迎关注以下平台，获取更多大模型求职面试资料、面经、项目：

- 📕 **小红书**：不转到大模型不改名
- 📺 **B站**：骑猪撞宝马71
