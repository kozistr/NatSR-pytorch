import torch.nn as nn


class Fractal(nn.Module):
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        pass


class NMD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
