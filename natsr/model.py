import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(
        self, n_feats: int = 64, nb_layers: int = 4, scale: float = 0.1
    ):
        super().__init__()
        self.n_feats = n_feats
        self.nb_layers = nb_layers
        self.scale = scale

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(self.n_feats * (i + 1), self.n_feats, kernel_size=3)
                for i in range(self.nb_layers - 1)
            ]
        )
        self.act = nn.ReLU()
        self.fusion_conv = nn.Conv2d(
            self.n_feats * (self.nb_layers - 1), self.n_feats, kernel_size=1
        )

        self._weight_initialize()

    def _weight_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode='fan_in', nonlinearity='relu'
                )

    def forward(self, x):
        concat_layers = [x]
        for i in range(self.nb_layers - 1):
            conv_concats = torch.cat(concat_layers, dim=1)
            _x = self.convs[i](conv_concats)
            _x = self.act(_x)
            concat_layers.append(_x)

        _x = torch.cat(concat_layers, dim=1)
        _x = self.fusion_conv(_x)

        return x + self.scale * _x


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
