import os
from typing import Dict

import torch
import torch.nn as nn
import yaml


def get_config(filename: str):
    with open(filename, 'r', encoding='utf8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


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
