import torch.nn as nn
from torch.optim import SGD, Adam

from natsr import OptimizerType


def build_optimizers(config, model: nn.Module):
    optimizer_type: str = config['model']['optimizer']

    if optimizer_type == OptimizerType.ADAM:
        optimizer = Adam(
            model.parameters(), lr=config['model']['lr'], amsgrad=False
        )
    elif optimizer_type == OptimizerType.SGD:
        optimizer = SGD(
            model.parameters(), lr=config['model']['lr'], nesterov=True
        )
    else:
        raise NotImplementedError(
            f'[-] not supported optimizer_type : {optimizer_type}'
        )

    return optimizer
