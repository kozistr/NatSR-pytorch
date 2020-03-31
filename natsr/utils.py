from typing import Dict

import yaml


def get_config(filename: str):
    with open(filename, 'r', encoding='utf8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def is_valid_key(d: Dict[str, str], key: str) -> bool:
    return key in d
