import os
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image
from math import sqrt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    RandomCrop,
    Resize,
    ToPILImage,
    ToTensor,
)

from natsr import DataSets, DataType, ModelType
from natsr.utils import is_gpu_available, is_valid_key


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


class DIV2KDataSet(Dataset):
    def __init__(self, config, data_type: str):
        self.config = config
        self.scale_factor: int = get_scale_factor(
            config['data'][data_type]['scale']
        )
        self.crop_size: int = get_valid_crop_size(
            config['model'][ModelType.NATSR]['height'], self.scale_factor
        )

        self.hr_image_paths: List[str] = []
        self.hr_images: np.ndarray = np.array([], dtype=np.uint8)

        self.hr_transform = hr_transform(self.crop_size)
        self.lr_transform = lr_transform(self.crop_size, self.scale_factor)

        self._get_image_paths(data_type=data_type)

    def _get_image_paths(self, data_type: str = DataType.TRAIN) -> None:
        dataset_type: str = self.config['data']['dataset_type']
        scale: int = self.config['data']['div2k']['scale']
        interp: str = self.config['data']['div2k']['interpolation']

        if not is_valid_key(self.config['data'], dataset_type):
            raise NotImplementedError(
                f'[-] not supported dataset_type : {dataset_type}'
            )

        dataset_path: str = self.config['data'][dataset_type]['dataset_path']
        if os.path.exists(dataset_path):
            self.hr_image_paths = sorted(
                glob(
                    os.path.join(
                        dataset_path,
                        f'DIV2K_{data_type}_HR_{interp}',
                        f'X{scale}',
                        '*.png',
                    )
                )
            )
        else:
            raise FileNotFoundError(
                f'[-] there\'s no dataset at {dataset_path}'
            )

    def __getitem__(self, index: int):
        hr_image = self.hr_transform(Image.open(self.hr_image_paths[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_image_paths)


def build_data_loader(config, data_type: str) -> DataLoader:
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
        batch_size=config['model'][model_type]['batch_size'],
        shuffle=True,
        pin_memory=is_gpu_available(),
        drop_last=False,
        num_workers=config['aux']['n_threads'],
    )

    return data_loader


def build_loader(config) -> Tuple[DataLoader, DataLoader]:
    train_data_loader = build_data_loader(config, data_type=DataType.TRAIN)
    valid_data_loader = build_data_loader(config, data_type=DataType.VALID)
    return train_data_loader, valid_data_loader
