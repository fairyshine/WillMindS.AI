<p align="center">
  <a href="https://fairyshine.github.io/WillMindS.AI/"><img src="https://github.com/fairyshine/WillMindS.AI/blob/master/icon.png?raw=true" alt="WillMindS" style="width: 30%;"></a>
</p>

# WillMindS

Efficiently set up your experimental environment with features such as logging, parameter management, code backup, and algorithms.

WillMindS is portable and designed primarily for personal use, serving as a helpful reference.

## How to use

Follow these steps to integrate WillMindS into your project:

1. **Copy the `WillMindS` directory:** Move the WillMindS directory into your `src/` path.

2. **Install dependencies:** Run the following command to install the required packages:

```shell
pip install -r src/WillMindS/requirements.txt
```

3. **Create configuration directory:** Make a new directory named `config/`, and create a file `basic.yaml` based on the provided template(in config_template).

4. **Update your `src/main.py`**: Add the following code to your main Python file:

```Python
from WillMindS import config, logger
from WillMindS.utils import backup_files

def main():
    # get config
    exp_name = config.experiment
    # logging
    logger.info("test the log")
    pass

if __name__ == "__main__":
		import os
		backup_files("src/",  # backup your code
               ["src/WillMindS"], # exclude this dir
                 os.path.join(config.output_dir,"source_code_backup")) # backup path
		main()
```

```shell
python src/main.py --config_file config/basic.yaml
```

