#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import random
from dataclasses import fields

from omegaconf import OmegaConf
import numpy as np
import torch

# cli_args = OmegaConf.from_cli()
# file_config = OmegaConf.load(cli_args.config)
# del cli_args.config

# default_config = OmegaConf.structured(PPOConfig())
# ppo_config = OmegaConf.merge(default_config, file_config, cli_args)
# ppo_config = OmegaConf.to_object(ppo_config)

def get_config():
    print('------------------------------------')
    print('------------读取Config中------------')
    cli_args = OmegaConf.from_cli()
    cli_args.config = cli_args.get("config","config/default.yaml")
    file_args = OmegaConf.load(cli_args.config)
    del cli_args.config

    config = OmegaConf.merge(file_args, cli_args)
    # config = OmegaConf.to_object(config)

    if config.model_name is None:
        config.model_name = 'model'
    return config

def load_arguments(data_class, config_dict):
    return data_class(**{k: v for k, v in config_dict.items() if k in {f.name for f in fields(data_class)}})

def set_seed(seed=1116):
    os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # set seed for cpu
    torch.cuda.manual_seed(seed)  # set seed for current gpu
    torch.cuda.manual_seed_all(seed)  # set seed for all gpu
    torch.backends.cudnn.deterministic = True

def init_output_dir(config):
    config.output_dir = os.path.join(config.output_total_dir, config.experiment+" | "+config.model_name+" | "+config.time+'/')
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    config.train.output_dir = os.path.join(config.output_dir, "train")
    if not os.path.exists(config.train.output_dir):
        os.makedirs(config.train.output_dir)

def backup_config(config):
    config_backup_path = os.path.join(config.output_dir, 'config.yaml')
    with open(config_backup_path, 'w', encoding='utf-8') as fw:
        OmegaConf.save(config=config, f=fw)

def print_config(config):
    print(OmegaConf.to_yaml(config))

def log_print_config(config,logger):
    logger.info('------------------------------------')
    logger.info('Here\'s the config:')
    logger.info(OmegaConf.to_yaml(config))


# class Config(object):
#     def __init__(self):
#         # 初始化设置
#         args = self.__get_config()

#         #将设置逐条引入类的实例变量
#         print('------------------------------------')
#         print('-------------读取超参数中-------------')
#         ''' argparse引入
#         for key in args.__dict__:
#             print('key: ',key)
#             setattr(self, key, args.__dict__[key])
#         '''
#         # omegeconf引入
#         for key in args:
#             print('key: ',key)
#             setattr(self, key, args.__getattr__(key))

#         self.start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # OR  str(datetime.datetime.now())[:-7]

#         #####检测可使用的运算设备#####
#         # if self.device_wanted=='cuda' and torch.cuda.is_available():
#         #     CHIP="cuda"   #Nvidia - Compute Unified Device Architecture
#         # elif self.device_wanted=='mps' and torch.backends.mps.is_built():
#         #     CHIP="mps"    #Apple Silicon - API Metal - Metal Performance Shaders
#         # else:
#         #     CHIP="cpu"
#         # 选择运算设备
#         # self.device = None
#         # if self.device_id >= 0 and CHIP != "cpu":
#         #     self.device = torch.device('{}:{}'.format(CHIP,self.device_id))
#         # else:pip 
#         #     self.device = torch.device(CHIP)

#         # 辅助处理 model_name 和 output_dir
#         if self.model_name is None:
#             self.model_name = 'model'
#         self.output_dir = os.path.join(self.output_total_dir, self.experiment+" | "+self.model_name+" | "+self.start_time+'/')
#         if not os.path.exists(self.output_dir):
#             os.makedirs(self.output_dir)

#         # 备份设置
#         self.__config_backup(args)

#         # 设置各类随机数
#         self.__set_seed(self.seed)

#     def __get_config(self):
#         parser = argparse.ArgumentParser()
#         parser.description = 'Config-setting filedir to set. All settings are in it.'
#         parser.add_argument("--config", type=str, default="config/basic.yaml",
#                         help="the filepath of config")
#         args = parser.parse_args()

#         args_dict=dict()
#         for key in args.__dict__:
#             args_dict[key]=args.__dict__[key]
#         cli_conf=OmegaConf.create(args_dict)
#         file_conf=OmegaConf.load(args.config)
#         return OmegaConf.merge(cli_conf,file_conf)

#     def __set_seed(self, seed=1116):
#         os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)  # set seed for cpu
#         torch.cuda.manual_seed(seed)  # set seed for current gpu
#         torch.cuda.manual_seed_all(seed)  # set seed for all gpu
#         torch.backends.cudnn.deterministic = True

#     def __config_backup(self, args):
#         config_backup_path = os.path.join(self.output_dir, 'config.yaml')
#         with open(config_backup_path, 'w', encoding='utf-8') as fw:
#             OmegaConf.save(config=args, f=fw)

#     def print_config(self):
#         for key in self.__dict__:
#             print(key, end=' = ')
#             print(self.__dict__[key])

#     def log_print_config(self,logger):
#         logger.info('------------------------------------')
#         logger.info('Here\'s the config:')
#         for key in self.__dict__:
#             logger.info(str(key)+' = '+str(self.__dict__[key]))

if __name__ == '__main__':
    config = get_config()
    print_config(config)

# yaml
# experiment: ...
# model_name: ...
# checkpoint: ...
# seed: 42
# output_total_dir: output/