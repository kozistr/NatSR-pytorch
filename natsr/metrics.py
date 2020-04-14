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
    full: bool = False,
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
        ssim_map = ssim_map.mean()
    else:
        ssim_map = ssim_map.mean(1).mean(1).mean(1)

    if not full:
        return ssim_map

    _v1 = 2.0 * sigma12 + c2
    _v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(_v1 / _v2)

    return ssim_map, cs


def ssim(
    img1,
    img2,
    window_size: int = 11,
    size_average: bool = True,
    full: bool = False,
):
    (_, channel, height, width) = img1.size()

    _window_size = min(window_size, height, width)
    window = create_window(_window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, full)


def msssim(
    img1,
    img2,
    window_size: int = 11,
    size_average: bool = True,
    full: bool = True,
):
    weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    if img1.is_cuda:
        weights = weights.cuda(img1.get_device())
    weights = weights.type_as(img1)

    levels = weights.size(0)

    ms_ssim, mcs = [], []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size, size_average, full=full)

        ms_ssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ms_ssim = torch.stack(ms_ssim)
    mcs = torch.stack(mcs)

    pow1 = mcs ** weights
    pow2 = ms_ssim ** weights
    mssim = torch.prod(pow1[:-1] * pow2[-1])
    return mssim


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100.0
    return 20.0 * torch.log10(1.0 / mse)


def acc(_pred, _true):
    eq = torch.eq(torch.gt(_pred, 0.5).float(), _true)
    return 100.0 * torch.mean(eq.float())
