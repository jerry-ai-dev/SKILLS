# PyTorch 入门课程大纲

共16课（含4次考试），分4个阶段。每个阶段：讲解课 → 代码练习 → 测验 → 阶段考试。

## 第一阶段：基础 (Lesson 1-4 + Exam 1)

### Lesson 1: Tensor 张量入门
- 什么是张量？与 NumPy 的关系
- 创建张量的多种方式
- 张量的基本运算（加减乘除、矩阵乘法）
- GPU 加速初体验（.to('cuda')）
- **练习**: 用张量实现简单的图片亮度调节

### Lesson 2: 自动求导 Autograd
- 什么是梯度？为什么需要求导？
- requires_grad 和计算图
- backward() 反向传播
- 梯度下降的直觉理解
- **练习**: 用 autograd 手动实现线性回归

### Lesson 3: nn.Module 搭建神经网络
- nn.Module 是什么？为什么用它？
- nn.Linear, nn.ReLU 等基本层
- forward() 方法
- 参数管理 parameters()
- **练习**: 搭建一个两层全连接网络识别手写数字

### Lesson 4: 训练循环 Training Loop
- Loss 函数（MSE, CrossEntropy）
- 优化器（SGD, Adam）
- 完整训练循环：forward → loss → backward → step
- 验证与评估
- **练习**: 完整训练 MNIST 手写数字分类器

### 📝 Exam 1: 基础篇阶段考试
- 覆盖 Lesson 1-4 全部内容
- 10 题，满分 100 分
- 考完批卷 → 评分 → 薄弱点诊断 → 错题讲解 → 针对性复习

## 第二阶段：实战进阶 (Lesson 5-7 + Exam 2)

### Lesson 5: 数据加载 Dataset & DataLoader
- Dataset 类的设计模式
- DataLoader 的 batch, shuffle, num_workers
- transforms 数据增强
- **练习**: 加载自定义图片数据集

### Lesson 6: 卷积神经网络 CNN
- 卷积操作的直觉理解
- Conv2d, MaxPool2d, BatchNorm
- 经典 CNN 架构 (LeNet → 简化 ResNet)
- **练习**: 用 CNN 实现 CIFAR-10 图片分类

### Lesson 7: 序列模型与词嵌入
- Embedding 层 —— 文字变数字
- RNN/LSTM 基本概念
- 为什么 RNN 有问题？引出 Attention
- **练习**: 简单的文本情感分析

### 📝 Exam 2: 实战进阶篇阶段考试
- 覆盖 Lesson 5-7 全部内容
- 10 题，满分 100 分
- 考完批卷 → 评分 → 薄弱点诊断 → 错题讲解 → 针对性复习

## 第三阶段：Attention 与 Transformer (Lesson 8-10 + Exam 3)

### Lesson 8: Attention 注意力机制
- 为什么需要 Attention？
- Query, Key, Value 的直觉理解
- Scaled Dot-Product Attention 公式与代码实现
- Self-Attention 可视化
- **练习**: 从零实现 Self-Attention

### Lesson 9: Transformer 架构
- Multi-Head Attention
- Position Encoding 位置编码
- Feed-Forward Network
- Layer Norm 和残差连接
- Encoder-Decoder 结构
- **练习**: 搭建一个 mini-Transformer

### Lesson 10: 从 Transformer 到 GPT
- Decoder-only 架构（GPT 系列）
- Causal Mask（因果掩码）
- Token 预测与生成
- **练习**: 搭建一个 mini-GPT 做字符级文本生成

### 📝 Exam 3: Attention 与 Transformer 阶段考试
- 覆盖 Lesson 8-10 全部内容
- 10 题，满分 100 分
- 考完批卷 → 评分 → 薄弱点诊断 → 错题讲解 → 针对性复习

## 第四阶段：现代 AI 实践 (Lesson 11-12 + Final Exam)

### Lesson 11: 预训练模型与 Hugging Face
- 为什么不从零训练？
- Hugging Face transformers 库
- 用预训练模型做文本分类、文本生成
- Tokenizer 的作用
- **练习**: 用 BERT 做情感分析 / 用 GPT-2 生成文本

### Lesson 12: 微调与 AI 前沿展望
- Fine-tuning 微调的概念
- LoRA 等高效微调技术简介
- RLHF 与对齐简介
- 多模态、Agent 等前沿方向
- **练习**: 微调一个小模型做特定任务
- **毕业项目**: 选择一个感兴趣的方向做小项目

### 🎓 Exam 4: 期末综合考试
- 覆盖 Lesson 1-12 全部内容
- 15 题，满分 150 分（换算百分制）
- 全链路考察：从 Tensor → GPT → 微调
- 考完批卷 → 评分 → 全面薄弱点诊断 → 逐题讲解 → 学习建议 → 颁发成绩单
