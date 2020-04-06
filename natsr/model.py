import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from natsr import ModelType


class ResidualDenseBlock(nn.Module):
    def __init__(
        self, n_feats: int = 64, nb_layers: int = 4, scale: float = 0.1
    ):
        super().__init__()
        self.n_feats = n_feats
        self.nb_layers = nb_layers
        self.scale = scale

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.n_feats * (i + 1),
                    self.n_feats,
                    kernel_size=3,
                    padding=1,
                )
                for i in range(self.nb_layers - 1)
            ]
        )
        self.act = nn.ReLU()
        self.fusion_conv = nn.Conv2d(
            self.n_feats * (self.nb_layers - 1),
            self.n_feats,
            kernel_size=1,
            padding=0,
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


class Generator(nn.Module):
    def __init__(self, config, scale: int = 4):
        super().__init__()
        self.config = config['model']

        self.scale = scale
        self.channel = self.config[ModelType.NATSR]['channel']
        self.n_feats = self.config[ModelType.NATSR]['n_feats']
        self.n_rep_rd_blocks = self.config[ModelType.NATSR][
            'n_rep_rd_blocks'
        ]
        self.n_rd_blocks = self.config[ModelType.NATSR]['n_rd_blocks']
        self.nb_layers = self.config[ModelType.NATSR]['nb_layers']

        self.head_conv = nn.Conv2d(self.channel, self.n_feats, kernel_size=3)
        self.tail_conv = nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3)
        self.rgb_conv = nn.Conv2d(
            self.n_feats, self.channel, kernel_size=3, padding=1
        )

        self.rd_blocks = nn.ModuleList(
            [
                ResidualDenseBlock(
                    self.n_feats, nb_layers=self.nb_layers, scale=self.scale,
                )
                for _ in range(self.n_rep_rd_blocks)
                for _ in range(self.n_rd_blocks)
            ]
        )

        if self.scale == 4:
            n_pix_shuffle_feats: int = self.n_feats * (self.scale // 2) * (
                    self.scale // 2
            )

            self.up_conv1 = nn.Conv2d(
                self.n_feats, n_pix_shuffle_feats, kernel_size=3, padding=1
            )
            self.up_conv2 = nn.Conv2d(
                n_pix_shuffle_feats,
                n_pix_shuffle_feats,
                kernel_size=3,
                padding=1,
            )
            self.pixel_shuffle = nn.PixelShuffle(self.scale // 2)
        else:
            self.up_conv1 = nn.Conv2d(
                self.n_feats,
                self.n_feats * self.scale * self.scale,
                kernel_size=3,
                padding=1,
            )
            self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, x):
        x_conv1 = self.head_conv(x)

        x = x_conv1
        for i in range(self.n_rep_rd_blocks):
            x = self.rd_blocks[i](x)
        x_conv2 = x + x_conv1

        x = x_conv2
        for i in range(self.n_rep_rd_blocks):
            x = self.rd_blocks[i](x)
        x_conv3 = x + x_conv2 + x_conv1

        x = x_conv3
        for i in range(self.n_rep_rd_blocks):
            x = self.rd_blocks[i](x)
        x_conv4 = x + x_conv3

        x = x_conv4
        for i in range(self.n_rep_rd_blocks):
            x = self.rd_blocks[i](x)
        x_conv5 = x + x_conv4 + x_conv3 + x_conv1

        x = x_conv5
        x = self.tail_conv(x)
        x = x + x_conv1

        if self.scale == 4:
            x = self.up_conv1(x)
            x = self.pixel_shuffle(x)

            x = self.up_conv2(x)
            x = self.pixel_shuffle(x)
        else:
            x = self.up_conv1(x)
            x = self.pixel_shuffle(x)

        x = self.rgb_conv(x)

        return x

    def __str__(self):
        return 'Generator'


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['model']

        self.channel = self.config[ModelType.NATSR]['channel']
        self.n_feats = self.config[ModelType.NATSR]['n_feats']

        self.conv1_1 = spectral_norm(
            nn.Conv2d(
                self.channel, self.n_feats * 1, kernel_size=3, padding=1,
            )
        )
        self.conv1_2 = spectral_norm(
            nn.Conv2d(
                self.n_feats * 1,
                self.n_feats * 1,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )

        self.conv2_1 = spectral_norm(
            nn.Conv2d(
                self.n_feats * 1, self.n_feats * 2, kernel_size=3, padding=1
            )
        )
        self.conv2_2 = spectral_norm(
            nn.Conv2d(
                self.n_feats * 2,
                self.n_feats * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )

        self.conv3_1 = spectral_norm(
            nn.Conv2d(
                self.n_feats * 2, self.n_feats * 4, kernel_size=3, padding=1
            )
        )
        self.conv3_2 = spectral_norm(
            nn.Conv2d(
                self.n_feats * 4,
                self.n_feats * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )

        self.conv4_1 = spectral_norm(
            nn.Conv2d(
                self.n_feats * 4, self.n_feats * 8, kernel_size=3, padding=1
            )
        )
        self.conv4_2 = spectral_norm(
            nn.Conv2d(
                self.n_feats * 8,
                self.n_feats * 8,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )

        self.conv5_1 = spectral_norm(
            nn.Conv2d(
                self.n_feats * 8, self.n_feats * 16, kernel_size=3, padding=1
            )
        )
        self.conv5_2 = spectral_norm(
            nn.Conv2d(
                self.n_feats * 16,
                self.n_feats * 16,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )

        self.conv6_1 = spectral_norm(
            nn.Conv2d(self.n_feats * 16, 1, kernel_size=3, padding=1)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        self._weight_initialize()

    def _weight_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.act(x)
        x = self.conv1_2(x)
        x = self.act(x)

        x = self.conv2_1(x)
        x = self.act(x)
        x = self.conv2_2(x)
        x = self.act(x)

        x = self.conv3_1(x)
        x = self.act(x)
        x = self.conv3_2(x)
        x = self.act(x)

        x = self.conv4_1(x)
        x = self.act(x)
        x = self.conv4_2(x)
        x = self.act(x)

        x = self.conv5_1(x)
        x = self.act(x)
        x = self.conv5_2(x)
        x = self.act(x)

        x = self.conv6_1(x)
        x = self.gap(x)

        return x

    def __str__(self):
        return 'Discriminator'


class NMD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config['model']

        self.channel = self.config[ModelType.NMD]['channel']
        self.n_feats = self.config[ModelType.NMD]['n_feats']

        self.conv1_1 = nn.Conv2d(
            self.channel, self.n_feats * 1, kernel_size=3, padding=1,
        )
        self.conv1_2 = nn.Conv2d(
            self.n_feats * 1, self.n_feats * 1, kernel_size=3, padding=1
        )

        self.conv2_1 = nn.Conv2d(
            self.n_feats * 1, self.n_feats * 2, kernel_size=3, padding=1
        )
        self.conv2_2 = nn.Conv2d(
            self.n_feats * 2, self.n_feats * 2, kernel_size=3, padding=1
        )

        self.conv3_1 = nn.Conv2d(
            self.n_feats * 2, self.n_feats * 4, kernel_size=3, padding=1
        )
        self.conv3_2 = nn.Conv2d(
            self.n_feats * 4, self.n_feats * 4, kernel_size=3, padding=1
        )

        self.conv4_1 = nn.Conv2d(
            self.n_feats * 4, self.n_feats * 8, kernel_size=3, padding=1
        )
        self.conv4_2 = nn.Conv2d(
            self.n_feats * 8, self.n_feats * 8, kernel_size=3, padding=1
        )

        self.conv5_1 = nn.Conv2d(self.n_feats * 8, 1, kernel_size=3, padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.act = nn.ReLU()

        self._weight_initialize()

    def _weight_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode='fan_in', nonlinearity='relu'
                )

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.act(x)
        x = self.conv1_2(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv2_1(x)
        x = self.act(x)
        x = self.conv2_2(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv3_1(x)
        x = self.act(x)
        x = self.conv3_2(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv4_1(x)
        x = self.act(x)
        x = self.conv4_2(x)
        x = self.act(x)
        x = self.max_pool(x)

        x = self.conv5_1(x)
        x = self.gap(x)

        return x

    def __str__(self):
        return 'NMD'


def build_model(config, model_type: str, device: str):
    if model_type == ModelType.NATSR:
        gen_network = Generator(config).to(device)
        disc_network = Discriminator(config).to(device)
        nmd_network = NMD(config).to(device)
        return gen_network, disc_network, nmd_network
    elif model_type == ModelType.NMD:
        nmd_network = NMD(config).to(device)
        return nmd_network
    else:
        raise NotImplementedError(
            f'[-] not supported model_type : {model_type}'
        )
