import math

import torch
from torch import optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import Trainer

from willminds import monitor, config, logger
from willminds.data.corpus.MiniMindVL_dataset import VLMDataset
from willminds.model.framework.MiniMindVL import MiniMindVLM, MiniMindVLConfig
from willminds.pipeline.MiniMind_trainer import compute_loss_func
from willminds.pipeline.MiniMind_trainer import VL_Trainer as MiniMind_Trainer

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
model_config = MiniMindVLConfig(**config.model)
model = MiniMindVLM(model_config, clip_model_path=config.clip_model_path).to(config.train.device)
_, preprocess = MiniMindVLM.get_vision_model(model_path=config.clip_model_path)

train_dataset = VLMDataset(config.train.train_data_path, config.train.train_image_path, tokenizer, preprocess=preprocess,
                        image_special_token=model_config.image_special_token,
                        max_length=config.train.max_seq_len)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.train.per_device_train_batch_size,
    pin_memory=True,
    drop_last=False,
    shuffle=False,
    num_workers=config.train.dataloader_num_workers,
    sampler=None
)

trainer = MiniMind_Trainer(model, train_loader)
trainer.train()