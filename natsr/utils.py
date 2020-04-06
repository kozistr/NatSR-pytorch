import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from natsr import DeviceType, ModelType
from tensorboardX import SummaryWriter


def get_config(filename: str):
    with open(filename, 'r', encoding='utf8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def initialize_seed(device: str, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == DeviceType.GPU:
        torch.cuda.manual_seed_all(seed)


def initialize_torch(config):
    device: str = config['aux']['device']

    initialize_seed(device, config['aux']['seed'])

    use_gpu: bool = (device == DeviceType.GPU)
    torch.backends.cudnn.deterministic = False if use_gpu else True
    torch.backends.cudnn.benchmark = True if use_gpu else False


def is_valid_key(d: Dict[str, str], key: str) -> bool:
    return key in d


def is_gpu_available() -> bool:
    return torch.cuda.is_available()


def load_model(filepath: str, model: nn.Module, device: str):
    epoch: int = 1
    ssim: float = 0.0

    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)

        try:
            model.load_state_dict(checkpoint['model'])
            epoch = checkpoint['epoch']
            ssim = checkpoint['ssim']
        except KeyError:
            model.load_state_dict(checkpoint)

        print(f'[+] model {str(model)} ({filepath}) loaded! epoch : {epoch}')
    else:
        print(f'[-] model {str(model)} ({filepath}) is not loaded :(')

    return epoch, ssim


def load_models(
    config,
    device: str,
    gen_network: Optional[nn.Module],
    disc_network: Optional[nn.Module],
    nmd_network: Optional[nn.Module],
) -> Tuple[int, float]:
    start_epochs, ssim = load_model(
        config['log']['checkpoint']['nmd_model_path'], nmd_network, device
    )

    if config['model']['model_type'] == ModelType.NATSR:
        start_epochs, ssim = load_model(
            config['log']['checkpoint']['gen_model_path'], gen_network, device
        )
        _, _ = load_model(
            config['log']['checkpoint']['disc_model_path'], disc_network, device
        )

    return start_epochs, ssim


def save_model(
    filepath: str, model: nn.Module, epoch: int, ssim: Optional[float]
):
    model_info = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    if ssim:
        model_info.update({'ssim': ssim})

    torch.save(model_info, filepath)


def build_summary_writer(config):
    log_dir: str = config['log']['log_dir']
    model_type: str = config['model']['model_type']
    return SummaryWriter(logdir=os.path.join(log_dir, model_type))


def log_summary(summary, data, global_step: int):
    for k, v in data.item():
        if k.startswith('loss') or k.startswith('aux'):
            summary.add_scalar(k, v, global_step)
        else:
            summary.add_image(k, v, global_step)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().data.numpy()
