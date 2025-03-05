
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from willminds import config, logger
from willminds.data.corpus.MiniMind_dataset import PretrainDataset
from willminds.model.framework.MiniMind import MiniMindLM
from willminds.pipeline.MiniMind_trainer import pretrain_trainer

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
model = MiniMindLM(config.model).to(config.train.device)

logger.info(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')

train_dataset = PretrainDataset(config.train.data_path, tokenizer, max_length=config.train.max_seq_len)
train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=config.train.num_workers,
            sampler=None
        )

trainer = pretrain_trainer(model, train_loader)
trainer.train()