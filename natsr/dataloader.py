import os
from glob import glob
from typing import List

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from natsr.utils import is_valid_key


class DIV2KDataSet(Dataset):
    def __init__(self, config, data_type: str = 'train'):
        self.config = config

        self.lr_image_paths: List[str] = []
        self.hr_image_paths: List[str] = []

        self.lr_images: np.ndarray = np.array([], dtype=np.uint8)
        self.hr_images: np.ndarray = np.array([], dtype=np.uint8)

        self._get_image_paths(data_type=data_type)
        self._load_images()

    def _get_image_paths(self, data_type: str = 'train') -> None:
        dataset_type: str = self.config['data']['dataset_type']
        scale: int = self.config['data']['div2k']['scale']
        interp: str = self.config['data']['div2k']['interpolation']

        if not is_valid_key(self.config['data'], dataset_type):
            raise NotImplementedError(
                f'[-] not supported dataset_type : {dataset_type}'
            )

        dataset_path: str = self.config['data'][dataset_type]['dataset_path']
        if os.path.exists(dataset_path):
            self.lr_image_paths = sorted(
                glob(
                    os.path.join(
                        dataset_path,
                        f'DIV2K_{data_type}_LR_{interp}',
                        f'X{scale}',
                        '*.png',
                    )
                )
            )
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

    def _load_images(self) -> None:
        self.lr_images = np.asarray(
            [
                cv2.imread(lr_image_path)[::-1]
                for lr_image_path in tqdm(self.lr_image_paths)
            ],
            dtype=np.uint8,
        )
        self.lr_images = np.asarray(
            [
                cv2.imread(hr_image_path)[::-1]
                for hr_image_path in tqdm(self.hr_image_paths)
            ],
            dtype=np.uint8,
        )

    def __getitem__(self, item: int):
        return self.lr_images[item], self.hr_images[item]

    def __len__(self):
        return len(self.lr_image_paths)


class ImageDataLoader(DataLoader):
    pass
