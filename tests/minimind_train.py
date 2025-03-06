import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from willminds import config, logger
from willminds.data.corpus.MiniMind_dataset import PretrainDataset, SFTDataset
from willminds.model.framework.MiniMind import MiniMindLM, LMConfig
from willminds.pipeline.MiniMind_trainer import Trainer

# -- pretrain --

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
model = MiniMindLM(LMConfig(**config.model)).to(config.train.device)

logger.info(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

train_dataset = PretrainDataset(config.train.data_path, tokenizer, max_length=config.train.max_seq_len)
train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.per_device_train_batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=config.train.num_workers,
            sampler=None
        )

trainer = Trainer(model, train_loader)
trainer.train()

# -- SFT --

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

model = MiniMindLM(LMConfig(**config.model))
model.load_state_dict(torch.load(config.checkpoint, map_location=config.train.device), strict=False)
model = model.to(config.train.device)

logger.info(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

train_dataset = SFTDataset(config.train.data_path, tokenizer, max_length=config.train.max_seq_len)
train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.per_device_train_batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=config.train.num_workers,
            sampler=None
        )

trainer = Trainer(model, train_loader)
trainer.train()