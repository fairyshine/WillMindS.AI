import time
import math
from contextlib import nullcontext

import torch
from torch import optim, nn

from .. import monitor, logger, config

def compute_loss_func(outputs, labels, num_items_in_batch):
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    if not num_items_in_batch:
        num_items_in_batch = config.train.per_device_train_batch_size

    loss = loss_fct(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        labels.view(-1)
    ).view(labels.size())
    loss = (loss * outputs.loss_mask).sum() / outputs.loss_mask.sum()
    loss += outputs.aux_loss
    loss /= config.train.gradient_accumulation_steps
    return loss

class Trainer:
    def __init__(self, model, train_loader):

        self.model = model

        self.config = config
        self.train_args = config.train

        self.train_loader = train_loader

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.train_args.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.train_args.learning_rate)

    def train(self):
        for epoch in range(self.train_args.num_train_epochs):
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
        for step, data_batch in enumerate(self.train_loader):
            data_batch["input_ids"] = data_batch["input_ids"].to(self.train_args.device)
            data_batch["label_ids"] = data_batch["label_ids"].to(self.train_args.device)
            data_batch["loss_mask"] = data_batch["loss_mask"].to(self.train_args.device)

            lr = self.get_lr(epoch * iter_per_epoch + step, self.train_args.num_train_epochs * iter_per_epoch, self.train_args.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                res = self.model(**data_batch)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    data_batch["label_ids"].view(-1)
                ).view(data_batch["label_ids"].size())
                loss = (loss * res.loss_mask).sum() / res.loss_mask.sum()
                loss += res.aux_loss
                loss = loss / self.train_args.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.train_args.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_args.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

            if step % self.train_args.logging_steps == 0:
                spend_time = time.time() - start_time
                logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                        epoch + 1,
                        self.train_args.num_train_epochs,
                        step,
                        iter_per_epoch,
                        loss.item() * self.train_args.gradient_accumulation_steps,
                        self.optimizer.param_groups[-1]['lr'],
                        spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

                if monitor.tracking:
                    monitor.tracking.log({"loss": loss.item() * self.train_args.gradient_accumulation_steps,
                        "lr": self.optimizer.param_groups[-1]['lr'],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

            if (step + 1) % self.train_args.save_steps == 0:
                self.model.eval()
                moe_path = '_moe' if self.model.config.use_moe else ''
                ckp = f'{self.train_args.output_dir}/pretrain_{self.model.config.dim}{moe_path}/'
                self.model.save_pretrained(ckp, safe_serialization=False)

                self.model.train()

class VL_Trainer:
    def __init__(self, model, train_loader):
        
        self.model = model

        self.config = config
        self.train_args = config.train

        self.train_loader = train_loader

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.train_args.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.train_args.learning_rate)

    @staticmethod
    def get_lr(current_step, total_steps, lr):
        return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

    def train(self):
        for epoch in range(self.train_args.num_train_epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):
        iter_per_epoch = len(self.train_loader)
        device_type = "cuda" if "cuda" in self.train_args.device else "cpu"
        ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        start_time = time.time()
        for step, data_batch in enumerate(self.train_loader):
            # X = X.to(self.train_args.device)
            # Y = Y.to(self.train_args.device)
            # loss_mask = loss_mask.to(self.train_args.device)
            # pixel_tensors = pixel_tensors.to(self.train_args.device)

            data_batch["input_ids"] = data_batch["input_ids"].to(self.train_args.device)
            data_batch["label_ids"] = data_batch["label_ids"].to(self.train_args.device)
            data_batch["loss_mask"] = data_batch["loss_mask"].to(self.train_args.device)
            data_batch["pixel_tensors"] = data_batch["pixel_tensors"].to(self.train_args.device)
            lr = self.get_lr(epoch * iter_per_epoch + step, self.train_args.num_train_epochs * iter_per_epoch, self.train_args.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            with ctx:
                res = self.model(**data_batch)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    data_batch["label_ids"].view(-1)
                ).view(data_batch["label_ids"].size())

                loss = (loss * res.loss_mask).sum() / res.loss_mask.sum()
                loss += res.aux_loss
                loss = loss / self.train_args.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.train_args.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_args.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

            if step % self.train_args.logging_steps == 0:
                spend_time = time.time() - start_time
                logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch + 1,
                        self.train_args.num_train_epochs,
                        step,
                        iter_per_epoch,
                        loss.item(),
                        self.optimizer.param_groups[-1]['lr'],
                        spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

                if monitor.tracking:
                    monitor.tracking.log({"loss": loss,
                            "lr": self.optimizer.param_groups[-1]['lr'],
                            "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

            if (step + 1) % self.train_args.save_steps == 0:
                self.model.eval()
                moe_path = '_moe' if self.model.config.use_moe else ''
                ckp = f'{self.train_args.output_dir}/pretrain_vlm_{self.model.config.dim}{moe_path}/'
                self.model.save_pretrained(ckp, safe_serialization=False)
                
                self.model.train()