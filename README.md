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
  <h3>"Explore to building the intelligence easily!"</h3>
</div>

<div align="center">

English | [中文](./README_CN.md)

</div>

## How to use

WillMindS provides a comprehensive framework for building and training AI models with built-in monitoring, configuration management, and various neural network components.

### Installation

Install the package using pip:

```shell
pip install willminds
```

### Basic Usage

1. **Import and Initialize**

```python
import willminds

# Access configuration and logger
config = willminds.config
logger = willminds.logger

# On first run, config will be automatically generated in "config/default.yaml"
```

2. **Configuration Management**

WillMindS automatically handles configuration with YAML files. You can:

- Use the default configuration: `config/default.yaml`
- Create custom configurations: `config/custom.yaml`
- Override specific settings at runtime

Example configuration structure:
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

3. **Basic Example**

```python
from willminds import config, logger
from willminds.utils import backup_files
import os

def main():
    # Access configuration
    exp_name = config.experiment

    # Logging
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Learning rate: {config.train.learning_rate}")

    # Backup code
    backup_files("src/",
                ["src/test"],
                os.path.join(config.output_total_dir, "src_backup"))

    # Your training logic here
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
```

4. **Running with Custom Configuration**

```shell
python your_script.py config=config/custom.yaml
```

### Advanced Features

- **Built-in Monitoring**: Integration with SwanLab and WandB for experiment tracking
- **Model Components**: Various attention mechanisms, feed-forward networks, normalization layers
- **Training Pipelines**: Pre-built trainers for different datasets
- **Utilities**: File backup, parsing, and threading utilities

### Configuration Files

WillMindS supports multiple configuration files:

- `config/default.yaml`: Default configuration
- `config/minimum.yaml`: Minimal configuration for basic usage
- Custom YAML files for specific experiments

### Key Components

- **Monitor**: Central monitoring and logging system
- **Model Modules**: Attention, FeedForward, Normalization layers
- **Data Loaders**: Corpus and dataset management
- **Optimizers**: Loss functions and optimization strategies
- **Pipelines**: Training and evaluation workflows

