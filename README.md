<p align="center">
  <a href="https://fairyshine.github.io/WillMindS.AI/"><img src="https://github.com/fairyshine/WillMindS.AI/blob/master/icon.png?raw=true" alt="WillMindS" style="width: 30%;"></a>
</p>

# WillMindS

Explore to building the intelligence easily! It's based on Torch, Transformers, etc.

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

