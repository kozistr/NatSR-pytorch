__REFERENCES__ = [
    'https://github.com/leftthomas/SRGAN/blob/master/pytorch_ssim/__init__.py'
]

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def gaussian(window_size: int, sigma: int) -> torch.Tensor:
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1d_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2d_window = (
        _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    )
    window = Variable(
        _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(
    img1,
    img2,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size: int = 11, size_average: bool = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
