import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


def instantiate_scheduler(optimizer, total_epochs, min_lr, opt_scheduler="WarmupCosineScheduler"):
    if opt_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=min_lr
        )
    elif opt_scheduler == "WarmupCosineScheduler":
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=total_epochs//10, total_epochs=total_epochs, min_lr=min_lr
        )
    else:
        raise ValueError(f"Got {opt_scheduler}")
    return scheduler



class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        """
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of epochs for warmup
            total_epochs: Total number of epochs
            min_lr: Minimum learning rate
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            alpha = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * alpha for base_lr in self.base_lrs]
