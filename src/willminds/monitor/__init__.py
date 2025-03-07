import datetime

from dataclasses import fields
from transformers import TrainingArguments

from .config import (
    get_config, 
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
        
        self.logger = Logger(self.config)
        self.tracking, self.tracking_callback = get_tracking(self.config)

        init_output_dir(self.config)
        set_seed(self.config.train.seed)

        log_print_config(self.config, self.logger)

        self.trainer_args = TrainingArguments(TrainingArguments(**{k: v for k, v in self.config.train.items() if k in {f.name for f in fields(TrainingArguments)}}))
    


