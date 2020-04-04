import torch.nn as nn
from torch.optim import SGD, Adam

from natsr import OptimizerType
from natsr.utils import is_valid_key


def build_optimizers(config, model: nn.Module):
    optimizer_type: str = config['model']['optimizer']
    model_type: str = config['model']['model_type']

    if not is_valid_key(config['model'], model_type):
        raise NotImplementedError(
            f'[-] not supported model_type : {model_type}'
        )

    if optimizer_type == OptimizerType.ADAM:
        optimizer = Adam(
            model.parameters(),
            lr=config['model'][model_type]['lr'],
            amsgrad=False,
        )
    elif optimizer_type == OptimizerType.SGD:
        optimizer = SGD(
            model.parameters(),
            lr=config['model'][model_type]['lr'],
            nesterov=True,
        )
    else:
        raise NotImplementedError(
            f'[-] not supported optimizer_type : {optimizer_type}'
        )

    return optimizer
