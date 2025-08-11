import datetime
import importlib
from dataclasses import fields

from .config import (
    get_config, 
    load_arguments,
    set_seed, 
    init_output_dir,
    backup_config, 
    print_config, 
    log_print_config)
from .logging import Logger
from .tracking import get_tracking


class Monitor:
    def __init__(self):
        self.time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # OR  str(datetime.datetime.now())[:-7]
        
        self.config = get_config()
        self.config.time = self.time
        set_seed(self.config.train.seed)
        if self.config.save_log:
            init_output_dir(self.config)


        self.logger = Logger(self.config)
        self.tracking, self.tracking_callback = get_tracking(self.config)


        log_print_config(self.config, self.logger)
        if self.config.train.load_training_arguments:
            if importlib.util.find_spec("transformers") and importlib.util.find_spec("accelerate"):
                from transformers import TrainingArguments
                self.logger.info("-"*36)
                self.logger.info("加载 config.train 中的 Huggingface Trainer 设置~")
                self.logger.info("-"*36)
                self.trainer_args = TrainingArguments(**{k: v for k, v in self.config.train.items() if k in {f.name for f in fields(TrainingArguments)}})