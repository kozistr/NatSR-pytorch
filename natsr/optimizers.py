import torch.nn as nn
from torch.optim import SGD, Adam

from natsr import OptimizerType


def build_optimizers(config, model_type: str, model: nn.Module):
    optimizer_type: str = config['model'][model_type]['optimizer']

    if optimizer_type == OptimizerType.ADAM:
        return Adam(
            model.parameters(),
            lr=float(config['model'][model_type]['lr']),
            amsgrad=False,
        )
    if optimizer_type == OptimizerType.SGD:
        return SGD(
            model.parameters(),
            lr=float(config['model'][model_type]['lr']),
            nesterov=True,
        )
    raise NotImplementedError(
        f'[-] not supported optimizer_type : {optimizer_type}'
    )
