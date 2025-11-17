<p align="center">
  <a href="https://fairyshine.github.io/WillMindS.AI/"><img src="https://github.com/fairyshine/WillMindS.AI/blob/master/icon.png?raw=true" alt="WillMindS" style="width: 30%;"></a>
</p>

<div align="center">

[![GitHub Repo stars](https://img.shields.io/github/stars/fairyshine/WillMindS.AI?style=social)](https://github.com/fairyshine/WillMindS.AI/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/fairyshine/WillMindS.AI)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/fairyshine/WillMindS.AI)](https://github.com/fairyshine/WillMindS.AI/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/fairyshine/WillMindS.AI/pulls)

</div>

<div align="center">
  <h3>"轻松探索并构建智能！"</h3>
</div>


<div align="center">

[English](./README.md) | 中文

</div>

## 如何使用

WillMindS 提供了一个全面的框架，用于构建和训练 AI 模型，具有内置监控、配置管理和各种神经网络组件。

### 安装

使用 pip 安装包：

```shell
pip install willminds
```

### 基本用法

1. **导入和初始化**

```python
import willminds

# 访问配置和日志记录器
config = willminds.config
logger = willminds.logger

# 首次运行时，配置将自动在 "config/default.yaml" 中生成
```

2. **配置管理**

WillMindS 自动处理 YAML 文件的配置。您可以：

- 使用默认配置：`config/default.yaml`
- 创建自定义配置：`config/custom.yaml`
- 在运行时覆盖特定设置

配置结构示例：
```yaml
output_total_dir: output/
experiment: main
model_name: model

tracking:
  type: swanlab
  project: WillMindS.AI
  mode: cloud

train:
  learning_rate: 5e-4
  num_train_epochs: 1
  per_device_train_batch_size: 32

model:
  dim: 512
  n_layers: 8
  n_heads: 8
  vocab_size: 6400
```

3. **基础示例**

```python
from willminds import config, logger
from willminds.utils import backup_files
import os

def main():
    # 访问配置
    exp_name = config.experiment

    # 日志记录
    logger.info(f"开始实验: {exp_name}")
    logger.info(f"学习率: {config.train.learning_rate}")

    # 备份代码
    backup_files("src/",
                ["src/test"],
                os.path.join(config.output_total_dir, "src_backup"))

    # 您的训练逻辑在这里
    logger.info("训练成功完成")

if __name__ == "__main__":
    main()
```

4. **使用自定义配置运行**

```shell
python your_script.py config=config/custom.yaml
```

### 高级功能

- **内置监控**：与 SwanLab 和 WandB 集成，用于实验跟踪
- **模型组件**：各种注意力机制、前馈网络、归一化层
- **训练管道**：为不同数据集预构建的训练器
- **实用工具**：文件备份、解析和线程实用程序

### 配置文件

WillMindS 支持多种配置文件：

- `config/default.yaml`：默认配置
- `config/minimum.yaml`：基本使用的最小配置
- 用于特定实验的自定义 YAML 文件

### 关键组件

- **Monitor**：中央监控和日志记录系统
- **Model Modules**：注意力、前馈、归一化层
- **Data Loaders**：语料库和数据集管理
- **Optimizers**：损失函数和优化策略
- **Pipelines**：训练和评估工作流程