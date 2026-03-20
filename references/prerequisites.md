# 各课前置知识表

老师在 Step 2（前置知识摸底）中使用。开课后展示对应课程的表格，让学生自选需要讲解的概念。

> 🎬 = 推荐配合视频辅助理解（详见 [video_guide.md](video_guide.md)）

---

## Lesson 1: Tensor 张量入门
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 标量、向量、矩阵 (Scalar/Vector/Matrix) | ⭐ | 数学中不同维度的数据表示方式 | |
| NumPy 数组 | ⭐ | Python 中最常用的数值计算库的核心数据结构 | |
| 数据类型 (dtype: float32, int64) | ⭐ | 计算机存储不同精度数字的方式 | |
| GPU 与并行计算 | ⭐⭐ | 为什么显卡能加速深度学习 | 🎬 |
| 广播机制 (Broadcasting) | ⭐⭐ | 不同形状的数组如何自动对齐运算 | 🎬 |

## Lesson 2: 自动求导 Autograd
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 导数 / 微分 (Derivative) | ⭐⭐ | 函数在某个点的变化率，"曲线的斜率" | 🎬 |
| 偏导数 (Partial Derivative) | ⭐⭐ | 多变量函数对其中一个变量求导 | 🎬 |
| 链式法则 (Chain Rule) | ⭐⭐⭐ | 复合函数求导的核心规则，反向传播的数学基础 | 🎬 |
| 梯度 (Gradient) | ⭐⭐ | 所有偏导数组成的向量，指向函数增长最快的方向 | 🎬 |
| 梯度下降 (Gradient Descent) | ⭐⭐ | 沿梯度反方向更新参数来最小化损失 | 🎬 |
| 学习率 (Learning Rate) | ⭐ | 每次更新参数时迈出的步长大小 | |
| 线性回归 (Linear Regression) | ⭐⭐ | 用直线 y = wx + b 拟合数据的最基础机器学习模型 | 🎬 |
| 损失函数 / 均方误差 (MSE Loss) | ⭐⭐ | 衡量预测值与真实值差距的指标 | |
| 计算图 (Computational Graph) | ⭐⭐⭐ | PyTorch 记录运算过程的 DAG 图，用于自动求导 | 🎬 |

## Lesson 3: nn.Module 搭建神经网络
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 神经元 / 感知机 (Perceptron) | ⭐⭐ | 模仿生物神经元的最小计算单元 | 🎬 |
| 激活函数 (Activation Function) | ⭐⭐ | 给神经网络引入非线性的函数（ReLU, Sigmoid等） | 🎬 |
| 全连接层 (Fully Connected / Linear Layer) | ⭐⭐ | 每个输入与每个输出都有连接权重的层 | |
| 前向传播 (Forward Pass) | ⭐⭐ | 数据从输入层流向输出层的计算过程 | |
| 反向传播 (Backpropagation) | ⭐⭐⭐ | 从输出到输入逐层计算梯度的算法 | 🎬 |
| 权重与偏置 (Weights & Bias) | ⭐ | 神经网络中需要学习的参数 | |
| 过拟合 / 欠拟合 (Overfitting / Underfitting) | ⭐⭐ | 模型学得"太好"或"不够好"的问题 | 🎬 |

## Lesson 4: 训练循环 Training Loop
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 交叉熵损失 (Cross-Entropy Loss) | ⭐⭐ | 分类任务中衡量预测概率分布与真实标签差距的函数 | 🎬 |
| Softmax 函数 | ⭐⭐ | 将一组数值转换为概率分布（总和为1） | 🎬 |
| 优化器 (Optimizer: SGD, Adam) | ⭐⭐ | 自动执行梯度下降更新参数的工具 | |
| Epoch / Batch / Iteration | ⭐ | 训练过程中的三层循环概念 | |
| 训练集 / 验证集 / 测试集 | ⭐ | 数据划分策略，防止模型作弊 | |
| MNIST 数据集 | ⭐ | 手写数字图片数据集，深度学习界的"Hello World" | |
| 准确率 (Accuracy) | ⭐ | 正确预测数占总预测数的比例，最直观的分类评估指标 | |

## Lesson 5: 数据加载 Dataset & DataLoader
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 迭代器 / 生成器 (Iterator / Generator) | ⭐⭐ | Python 中按需生产数据的模式 | |
| 数据增强 (Data Augmentation) | ⭐⭐ | 对训练数据做随机变换来增加多样性 | 🎬 |
| 归一化 (Normalization) | ⭐⭐ | 将数据缩放到统一范围以加速训练 | |
| 多进程并行 (Multiprocessing) | ⭐⭐ | 使用多个 CPU 核心同时加载数据 | |

## Lesson 6: 卷积神经网络 CNN
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 卷积 (Convolution) | ⭐⭐⭐ | 使用滑动小窗口提取局部特征的运算 | 🎬 |
| 特征图 / 通道 (Feature Map / Channel) | ⭐⭐ | 卷积输出的每一层，代表检测到的一种特征 | 🎬 |
| 池化 (Pooling) | ⭐⭐ | 缩小特征图尺寸同时保留重要信息 | |
| 感受野 (Receptive Field) | ⭐⭐⭐ | 输出的一个像素"看到"输入多大范围 | 🎬 |
| 批归一化 (Batch Normalization) | ⭐⭐⭐ | 每一层归一化加速训练、稳定梯度 | 🎬 |
| 残差连接 (Residual Connection / Skip Connection) | ⭐⭐⭐ | ResNet 的核心思想，解决深层网络退化问题 | 🎬 |

## Lesson 7: 序列模型与词嵌入
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 词嵌入 (Word Embedding) | ⭐⭐ | 将单词映射到密集向量空间 | 🎬 |
| 独热编码 (One-Hot Encoding) | ⭐ | 用 0/1 向量表示类别的最原始方式 | |
| 循环神经网络 (RNN) | ⭐⭐⭐ | 带"记忆"的网络，处理序列数据 | 🎬 |
| 梯度消失 / 梯度爆炸 (Vanishing / Exploding Gradient) | ⭐⭐⭐ | RNN 训练中梯度变得极小或极大的问题 | 🎬 |
| LSTM / GRU | ⭐⭐⭐ | 用门控机制解决梯度消失的改进 RNN | 🎬 |
| 时间步 (Time Step) | ⭐⭐ | 序列中每个位置对应的输入 | |

## Lesson 8: Attention 注意力机制
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| Seq2Seq 模型 | ⭐⭐ | 输入一个序列、输出一个序列的模型框架 | 🎬 |
| Query / Key / Value | ⭐⭐⭐ | Attention 中三个核心角色，类比"提问 / 索引 / 答案" | 🎬 |
| 点积 (Dot Product) | ⭐ | 两个向量逐元素相乘再求和，衡量相似度 | |
| Softmax 归一化 | ⭐⭐ | 将注意力分数转化为概率分布 | |
| Self-Attention | ⭐⭐⭐ | 序列内部每个位置关注其他所有位置 | 🎬 |
| 注意力权重可视化 | ⭐⭐ | 直观查看模型"在看哪里" | 🎬 |

## Lesson 9: Transformer 架构
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| Multi-Head Attention | ⭐⭐⭐ | 并行多组 Attention，捕捉不同角度的关系 | 🎬 |
| 位置编码 (Positional Encoding) | ⭐⭐⭐ | 注入顺序信息，因为 Attention 本身不区分位置 | 🎬 |
| 残差连接 + Layer Norm | ⭐⭐ | 稳定深层 Transformer 训练的两个关键技巧 | |
| Feed-Forward Network (FFN) | ⭐⭐ | Transformer 中每个位置独立的两层全连接 | |
| Encoder-Decoder 架构 | ⭐⭐⭐ | "理解→生成"的两阶段结构 | 🎬 |

## Lesson 10: 从 Transformer 到 GPT
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 自回归生成 (Autoregressive Generation) | ⭐⭐⭐ | 逐个 token 生成，每次用已生成的内容预测下一个 | 🎬 |
| 因果掩码 (Causal Mask) | ⭐⭐⭐ | 阻止模型"偷看未来"的三角矩阵 | 🎬 |
| Token 与 Tokenizer | ⭐⭐ | 文本切分成模型能处理的最小单元 | |
| 温度采样 (Temperature Sampling) | ⭐⭐ | 控制生成文本的随机性和创造性 | |
| Top-k / Top-p 采样 | ⭐⭐ | 限制候选 token 的采样策略 | |

## Lesson 11: 预训练模型与 Hugging Face
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 预训练 (Pre-training) | ⭐⭐ | 在大规模数据上训练通用模型 | 🎬 |
| 迁移学习 (Transfer Learning) | ⭐⭐ | 将已学知识迁移到新任务 | |
| BERT vs GPT | ⭐⭐ | 双向理解 vs 单向生成，两种主流预训练范式 | 🎬 |
| Tokenizer (BPE, WordPiece) | ⭐⭐⭐ | 将文本拆分成子词的不同算法 | 🎬 |
| Pipeline API | ⭐ | Hugging Face 提供的一行代码推理接口 | |

## Lesson 12: 微调与 AI 前沿展望
| 概念 | 难度 | 一句话简介 | 🎬 |
|------|------|-----------|-----|
| 微调 (Fine-tuning) | ⭐⭐ | 在预训练模型基础上用少量数据适配新任务 | |
| LoRA / QLoRA | ⭐⭐⭐ | 仅训练少量参数的高效微调技术 | 🎬 |
| RLHF (人类反馈强化学习) | ⭐⭐⭐ | 用人类偏好训练 AI 对齐人类价值观 | 🎬 |
| 多模态 (Multimodal) | ⭐⭐ | 同时处理文本、图像、音频等多种输入 | |
| AI Agent | ⭐⭐ | AI 自主规划和使用工具完成复杂任务 | 🎬 |
