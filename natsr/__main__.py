from natsr import CONFIG_FILENAME
from natsr.test import test
from natsr.train import train
from natsr.utils import get_config


def main():
    config = get_config(CONFIG_FILENAME)

    mode: str = config['mode']
    if mode == 'train':
        train(config)
    elif mode == 'test':
        test(config)
    else:
        raise NotImplementedError(f'[-] not supported mode : {mode}')


main()
