import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

from natsr import DeviceType


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


def save_model(filepath: str, model: nn.Module, epoch: int):
    torch.save({'model': model.state_dict(), 'epoch': epoch}, filepath)
