from .config import Config
from .log import Logger

config = Config()
logger = Logger(config)
config.log_print_config(logger)

'''Use Example - main.py
import sys
sys.dont_write_bytecode = True
import os

from WillMindS import config, logger
from WillMindS.utils import backup_files

def main():
    pass

if __name__ == "__main__":
    # backup_files("src/",["src/WillMindS"],os.path.join(config.output_dir,"source_code_backup"))
    backup_files("src/",[],os.path.join(config.output_dir,"source_code_backup"))
    main()

python src/main.py --config_file config/basic.yaml
'''
