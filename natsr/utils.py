import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from natsr import DeviceType, ModelType


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

    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)

        try:
            model.load_state_dict(checkpoint['model'])
            epoch = checkpoint['epoch']
        except KeyError:
            model.load_state_dict(checkpoint)

        print(f'[+] model {filepath} loaded! epoch : {epoch}')
    else:
        print(f'[-] model is not loaded :(')

    return epoch


def load_models(
    config,
    device: str,
    gen_network: Optional[nn.Module],
    disc_network: Optional[nn.Module],
    nmd_network: Optional[nn.Module],
) -> int:
    start_epochs = load_model(
        config['checkpoint']['nmd_model_path'], nmd_network, device
    )
    if config['model']['model_type'] == ModelType.NATSR:
        start_epochs = load_model(
            config['checkpoint']['gen_model_path'], gen_network, device
        )
        load_model(
            config['checkpoint']['disc_model_path'], disc_network, device
        )
    return start_epochs


def save_model(
    filepath: str, model: nn.Module, epoch: int, ssim_score: Optional[float]
):
    model_info = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    if ssim_score:
        model_info.update(
            {'ssim': ssim_score,}
        )

    torch.save(model_info, filepath)
