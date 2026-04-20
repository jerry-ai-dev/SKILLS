# 项目依赖库

本项目教学过程中需要以下 Python 库。首次授课时（进度为空）自动检查并安装。

## 必需依赖

| 库名 | 用途 | 安装命令 |
|------|------|---------|
| `torch` | PyTorch 核心库，教学主体 | `python -m pip install torch` |
| `matplotlib` | 绘制函数曲线、数据可视化、激活函数图等 | `python -m pip install matplotlib` |

## 环境检查脚本

首次授课时执行以下命令检查环境：

```bash
python -c "import torch; print(f'torch={torch.__version__}'); import matplotlib; print(f'matplotlib={matplotlib.__version__}')"
```

如果报错 `ModuleNotFoundError`，则在当前环境中安装缺失的库：

```bash
python -m pip install torch matplotlib
```

## 说明

- 依赖安装**仅在首次授课（进度为空）时检查**，后续课程跳过
- 使用 `python -m pip install` 而非 `pip install`，确保安装到当前 Python 解释器对应的环境中
