from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer

from natsr import LRSchedulerType


# TODO : implement staircase exp lr scheduling
def build_lr_scheduler(config, model_type: str, optimizer: Optimizer):
    lr_schedule_type: str = config['model'][model_type]['lr_schedule_type']

    if lr_schedule_type == LRSchedulerType.EXPONENTIAL:
        return ExponentialLR(
            optimizer, gamma=config['model'][model_type]['lr_decay_ratio'],
        )
    raise NotImplementedError(
        f'[-] not supported lr_schedule_type {lr_schedule_type}'
    )
