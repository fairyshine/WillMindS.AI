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

Follow these steps to integrate WillMindS into your project:

1. **Install:** Run the following command to install the package:

```shell
pip install willminds
```

2. **Create configuration directory:** Make a new directory named `config/`, and create a file `basic.yaml` based on the provided template(in config_template).

3. **Update your `src/main.py`**: Add the following code to your main Python file:

```Python
from willminds import config, logger
from willminds.utils import backup_files

def main():
    # get config
    exp_name = config.experiment
    # logging
    logger.info("test the log")
    pass

if __name__ == "__main__":
		import os
		backup_files("src/",  # backup your code
               ["src/test"], # exclude this dir
                 os.path.join(config.output_dir,"source_code_backup")) # backup path
		main()
```

```shell
python src/main.py --config_file config/basic.yaml
```

