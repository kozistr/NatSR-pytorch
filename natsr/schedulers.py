from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer

from natsr import LRSchedulerType
from natsr.utils import is_valid_key


def build_lr_scheduler(config, optimizer: Optimizer):
    lr_schedule_type: str = config['model']['lr_schedule_type']
    model_type: str = config['model']['model_type']

    if not is_valid_key(config['model'], model_type):
        raise NotImplementedError(
            f'[-] not supported model_type : {model_type}'
        )

    if lr_schedule_type == LRSchedulerType.EXPONENTIAL:
        return ExponentialLR(
            optimizer, gamma=config['model'][model_type]['lr_decay_ratio']
        )

    raise NotImplementedError(
        f'[-] not supported lr_schedule_type {lr_schedule_type}'
    )
