import time
import math
from contextlib import nullcontext

import torch
from torch import optim, nn

from .. import monitor, logger, config

class Trainer:
    def __init__(self, model, train_loader):

        self.model = model

        self.config = config
        self.train_args = config.train

        self.train_loader = train_loader

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.train_args.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.train_args.lr)

    def train(self):
        for epoch in range(self.train_args.epochs):
            self.train_epoch(epoch)

    @staticmethod
    def get_lr(current_step, total_steps, lr):
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


    def train_epoch(self, epoch):
        iter_per_epoch = len(self.train_loader)
        device_type = "cuda" if "cuda" in self.train_args.device else "cpu"
        ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        start_time = time.time()
        for step, (X, Y, loss_mask) in enumerate(self.train_loader):
            X = X.to(self.train_args.device)
            Y = Y.to(self.train_args.device)
            loss_mask = loss_mask.to(self.train_args.device)

            lr = self.get_lr(epoch * iter_per_epoch + step, self.train_args.epochs * iter_per_epoch, self.train_args.lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                res = self.model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += res.aux_loss
                loss = loss / self.train_args.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.train_args.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_args.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

            if step % self.train_args.log_interval == 0:
                spend_time = time.time() - start_time
                logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                        epoch + 1,
                        self.train_args.epochs,
                        step,
                        iter_per_epoch,
                        loss.item() * self.train_args.accumulation_steps,
                        self.optimizer.param_groups[-1]['lr'],
                        spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

                if monitor.tracking:
                    monitor.tracking.log({"loss": loss.item() * self.train_args.accumulation_steps,
                        "lr": self.optimizer.param_groups[-1]['lr'],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

            if (step + 1) % self.train_args.save_interval == 0:
                self.model.eval()
                moe_path = '_moe' if self.model.config.use_moe else ''
                ckp = f'{config.output_dir}/pretrain_{self.model.config.dim}{moe_path}/'
                self.model.save_pretrained(ckp, safe_serialization=False)

                self.model.train()


