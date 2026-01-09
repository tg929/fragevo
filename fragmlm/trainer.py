"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging
import os.path

from tqdm import tqdm
import numpy as np

import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_iters = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_iters = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataloader, valid_dataloader, config, tokenizer, device, rank):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.config = config
        self.tokenizer = tokenizer

        # take over whatever gpus are on the system
        self.device = device
        self.rank = rank
        self.writer = SummaryWriter(f'./log')

    def save_checkpoint(self, tmp=True):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        path = self.config.ckpt_path + '_tmp.pt' if tmp else self.config.ckpt_path + '.pt'
        logger.info("saving %s", path)
        torch.save(raw_model.state_dict(), path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_dataloader if is_train else self.valid_dataloader

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                # forward the model
                with torch.amp.autocast('cuda'):
                    with torch.set_grad_enabled(is_train):
                        logits, loss, _ = model(x, self.tokenizer, y)
                        loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if it < config.warmup_iters:
                            # linear warmup
                            lr_mult = float(it) / float(max(1, config.warmup_iters))
                        else:
                            # cosine learning rate decay
                            progress = float(it - config.warmup_iters) / float(
                                max(1, config.final_iters - config.warmup_iters))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    if self.rank == 0:
                        self.writer.add_scalar('Loss/step_train_loss', loss, it + epoch * len(loader))
                        self.writer.add_scalar('LearningRate', lr, it + epoch * len(loader))
                        if it % 10000 == 0:
                            # 保存权重
                            self.save_checkpoint(True)
                            # 保存模型的状态字典，包括权重和优化器状态
                            checkpoint = {
                                'epoch': epoch,  # 当前的epoch（可以根据需要提供epoch）
                                'iter': it,  # 当前的迭代次数
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scaler_state_dict': scaler.state_dict(),
                                'loss': loss.item(),
                                'losses': losses
                            }
                            checkpoint_filename = f"./weights/checkpoint_{it}.pth"
                            torch.save(checkpoint, checkpoint_filename)

                        pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            # 同步不同 GPU 的损失并计算平均值
            average_loss = torch.tensor(losses).to(self.device)
            average_loss = average_loss.mean()
            dist.all_reduce(average_loss, op=dist.ReduceOp.SUM)
            average_loss = average_loss / dist.get_world_size()

            if not is_train:
                test_loss = float(average_loss.item())
                logger.info("test loss: %f", test_loss)
                return test_loss

            return float(average_loss.item())

        for epoch in range(config.max_epochs):
            train_loss = run_epoch('train')
            # test_loss = run_epoch('test')

            if self.rank == 0:
                # self.writer.add_scalar('Loss/epoch_valid_loss', test_loss, epoch + 1)
                self.writer.add_scalar('Loss/epoch_train_loss', train_loss, epoch + 1)
                self.save_checkpoint(False)

        return None
