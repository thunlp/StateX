from transformers import Trainer, PreTrainedTokenizerBase
from ..arguments import Args
from torch import nn
import torch
from accelerate import Accelerator


class HFTrainer(Trainer):
    def __init__(
        self,
        args: Args,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.accelerator = Accelerator()
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )
