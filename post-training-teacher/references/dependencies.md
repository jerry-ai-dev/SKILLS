# 后训练理论深化 - 依赖库清单

## 核心依赖

```bash
pip install torch transformers trl datasets gymnasium
```

## 各课用到的库

| 库 | 版本建议 | 用途 | 首次使用 |
|----|---------|------|---------|
| `torch` | ≥2.0 | 所有 PyTorch 计算（已安装自阶段一）| Lesson 1 |
| `gymnasium` | ≥0.29 | CartPole 等 RL 环境（Lesson 2 选做）| Lesson 2 |
| `transformers` | ≥4.40 | 加载预训练模型、Tokenizer | Lesson 4 |
| `trl` | ≥0.9 | SFTTrainer、GRPOTrainer | Lesson 7 |
| `datasets` | ≥2.0 | HuggingFace 数据集加载 | Lesson 7 |
| `sympy` | 任意 | 数学答案验证（Lesson 10 选做）| Lesson 10 |

## 环境检查命令

```bash
python -c "
import torch; print(f'torch: {torch.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
try:
    import trl; print(f'trl: {trl.__version__}')
except ImportError:
    print('trl: NOT INSTALLED - run: pip install trl')
try:
    import gymnasium; print(f'gymnasium: {gymnasium.__version__}')
except ImportError:
    print('gymnasium: NOT INSTALLED - run: pip install gymnasium (选做，Lesson 2 用)')
try:
    import datasets; print(f'datasets: {datasets.__version__}')
except ImportError:
    print('datasets: NOT INSTALLED - run: pip install datasets')
"
```

## 注意事项

- `trl` 和 `datasets` 在 Lesson 7 之前可以不安装
- `gymnasium` 是选做依赖，Lesson 2 的 CartPole 练习才需要，理论课不需要
- 所有代码演示都会先检查库是否可用，未安装时跳过实际运行，仅展示代码结构
