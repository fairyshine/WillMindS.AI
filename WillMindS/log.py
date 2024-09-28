import os
import logging 

class Logger:
    def __init__(self, config):
        print('--------------初始化日志--------------')
        # 实例化一个 Logger 
        self.logger = logging.getLogger("WillMindS") #logger = logging.getLogger() 

        # 设置日志输出等级 
        self.logger.setLevel(logging.DEBUG) 

        # 设置日志输出格式 
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # 实例化写入日志文件的Handler
        if not os.path.exists("WillMindS_log/"):
            # 如果文件夹不存在，则创建文件夹
            os.makedirs("WillMindS_log/")
        file_handler = logging.FileHandler('WillMindS_log/{} | {} | {}.log'.format(config.experiment, config.model_name, config.start_time))
        file_handler.setFormatter(formatter) 

        # 实例化实时输出的Handler
        try:
            from rich.logging import RichHandler
            shell_handler = RichHandler(rich_tracebacks=True)
        except ImportError:
            shell_handler = logging.StreamHandler(stream=None)
            shell_handler.setFormatter(formatter) 

        # 添加到 logger 
        self.logger.addHandler(file_handler) 
        self.logger.addHandler(shell_handler)

        # 输出日志 
        # logger.debug('this is a debug message') 
        # logger.info('this is an info message') 
        # logger.warning('this is a warning message') 
        # logger.error('this is an error message') 
        # logger.critical('this is a critical message')

        # 日志的存储模块
        self.kv_inventory = dict()

        self.info('------------------------------------')
        self.info('------------日志初始化完成------------')

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def save_kv(self, key, value):
        self.kv_inventory[key] == value
        return value
