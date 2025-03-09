import math

import torch
from torch import optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import Trainer

from willminds import monitor, config, logger
from willminds.data.corpus.MiniMind_dataset import PretrainDataset, SFTDataset
from willminds.model.framework.MiniMind import MiniMindLM, MiniMindConfig
from willminds.pipeline.MiniMind_trainer import compute_loss_func
from willminds.pipeline.MiniMind_trainer import Trainer as MiniMind_Trainer


# tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
# # * Pretrain
# # model = MiniMindLM(MiniMindConfig(**config.model)).to(config.train.device)
# # train_dataset = PretrainDataset(config.train.train_data_path, tokenizer, max_length=config.train.max_seq_len)
# # * SFT
# model = MiniMindLM.from_pretrained(config.checkpoint)
# train_dataset = SFTDataset(config.train.train_data_path, tokenizer, max_length=config.train.max_seq_len)

# logger.info(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

# optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate)
# scheduler = optim.lr_scheduler.LambdaLR(
#     optimizer=optimizer, 
#     lr_lambda=lambda step : 0.1 + 0.5 * (1 + math.cos(math.pi*step/(config.train.num_train_epochs*len(train_dataset)/config.train.per_device_train_batch_size/config.train.gradient_accumulation_steps)))
# )

# trainer = Trainer(model=model, 
#                   args=monitor.trainer_args,
#                   train_dataset=train_dataset,
#                   compute_loss_func=compute_loss_func,
#                   callbacks=[monitor.tracking_callback],
#                   optimizers=(optimizer,scheduler))
# trainer.train()

# ---- RL ---------
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
model = MiniMindLM.from_pretrained(config.checkpoint)
dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir=config.train.output_dir, logging_steps=10)
trainer = GRPOTrainer(
    # model="Qwen/Qwen2-0.5B-Instruct",
    model = model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[monitor.tracking_callback]
)
trainer.train()


# # -- pretrain (RAW) --

# tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
# model = MiniMindLM(LMConfig(**config.model)).to(config.train.device)
# train_dataset = PretrainDataset(config.train.train_data_path, tokenizer, max_length=config.train.max_seq_len)
# logger.info(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

# train_loader = DataLoader(
#             train_dataset,
#             batch_size=config.train.per_device_train_batch_size,
#             pin_memory=True,
#             drop_last=False,
#             shuffle=False,
#             num_workers=config.train.dataloader_num_workers,
#             sampler=None
#         )

# trainer = MiniMind_Trainer(model, train_loader)
# trainer.train()
