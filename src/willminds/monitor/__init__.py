import datetime

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
        self.tracking = get_tracking(self.config)

        init_output_dir(self.config)
        set_seed(self.config.seed)
    


