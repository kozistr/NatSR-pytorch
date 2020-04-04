from typing import Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from natsr import DataType
from natsr.dataloader import build_data_loader
from natsr.model import NMD, Discriminator, Fractal


def build_loader(config) -> Tuple[DataLoader, DataLoader]:
    train_data_loader = build_data_loader(config, data_type=DataType.TRAIN)
    valid_data_loader = build_data_loader(config, data_type=DataType.VALID)
    return train_data_loader, valid_data_loader


def build_model(config) -> Tuple[nn.Module, nn.Module, nn.Module]:
    gen_network = Fractal(config)
    nmd_network = NMD(config)
    disc_network = Discriminator(config)
    return gen_network, nmd_network, disc_network


def train(config):
    train_loader, valid_loader = build_loader(config)
    gen_network, nmd_network, disc_network = build_model(config)
