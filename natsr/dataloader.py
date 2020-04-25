import os
import random
from glob import glob
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch import cat
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    RandomCrop,
    Resize,
    ToPILImage,
    ToTensor,
)
from torchvision.transforms.functional import rotate

from natsr import DataSets, DataType, Mode, ModelType
from natsr.utils import get_blurry, get_noisy, is_gpu_available, is_valid_key


def get_scale_factor(scale: int) -> int:
    if scale & (scale - 1):
        return int(sqrt(scale))
    return scale


def get_valid_crop_size(crop_size: int, scale: int) -> int:
    return crop_size - (crop_size % scale)


def hr_transform(crop_size: int):
    return Compose([RandomCrop(crop_size), ToTensor()])


def lr_transform(crop_size: int, scale: int):
    return Compose(
        [
            ToPILImage(),
            Resize(crop_size // scale, interpolation=Image.BICUBIC),
            ToTensor(),
        ]
    )


def get_nmd_data(img, scale: int, alpha: float, sigma: float, mode: str):
    batch_size: int = img.size(0)

    if mode == Mode.TRAIN:
        noisy_img = get_noisy(img[: batch_size // 4, :, :, :], sigma)
        blurry_img = get_blurry(
            img[batch_size // 4 : batch_size // 2, :, :, :], scale, alpha
        )
        clean_img = img[batch_size // 2 :, :, :, :]
    else:
        noisy_img = get_noisy(img, sigma)
        blurry_img = get_blurry(img, 4, alpha)
        clean_img = img
    return cat([noisy_img, blurry_img, clean_img], dim=0)


class DIV2KDataSet(Dataset):
    def __init__(self, config, data_type: str):
        self.config = config

        self.scale_factor: int = get_scale_factor(
            config['data'][DataSets.DIV2K]['scale']
        )
        self.crop_size: int = get_valid_crop_size(
            config['model'][ModelType.FRSR]['height'], self.scale_factor
        )

        self.hr_image_paths: List[str] = []
        self.hr_images: np.ndarray = np.array([], dtype=np.uint8)

        self.hr_transform = hr_transform(self.crop_size)
        self.lr_transform = lr_transform(self.crop_size, self.scale_factor)

        self._get_image_paths(data_type=data_type)

    def _get_image_paths(self, data_type: str) -> None:
        dataset_path: str = self.config['data'][DataSets.DIV2K]['dataset_path']

        if os.path.exists(dataset_path):
            self.hr_image_paths = sorted(
                glob(
                    os.path.join(
                        dataset_path, f'DIV2K_{data_type}_HR', '*.png'
                    )
                )
            )
        else:
            raise FileNotFoundError(
                f'[-] there\'s no dataset at {dataset_path}'
            )

    def __getitem__(self, index: int):
        hr_image = Image.open(self.hr_image_paths[index])
        hr_image = rotate(hr_image, random.choice([0, 90, 180, 270]))
        hr_image = self.hr_transform(hr_image)
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_paths)


def build_data_loader(
    config, data_type: str, override_batch_size: Optional[int] = None
) -> DataLoader:
    dataset_type: str = config['data']['dataset_type']
    model_type: str = config['model']['model_type']

    if not is_valid_key(config['model'], model_type):
        raise NotImplementedError(
            f'[-] not supported model_type : {model_type}'
        )

    if dataset_type == DataSets.DIV2K:
        dataset = DIV2KDataSet(config, data_type)
    else:
        raise NotImplementedError(
            f'[-] not supported dataset_type : {dataset_type}'
        )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['model'][model_type]['batch_size']
        if override_batch_size is None
        else override_batch_size,
        shuffle=True,
        pin_memory=is_gpu_available(),
        drop_last=False,
        num_workers=config['aux']['n_threads'],
    )

    return data_loader


def build_loader(
    config, override_batch_size: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    train_data_loader = build_data_loader(
        config, data_type=DataType.TRAIN.value
    )
    valid_data_loader = build_data_loader(
        config,
        data_type=DataType.VALID.value,
        override_batch_size=override_batch_size,
    )
    return train_data_loader, valid_data_loader
