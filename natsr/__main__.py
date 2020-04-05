from natsr import CONFIG_FILENAME, Mode
from natsr.inference import inference
from natsr.test import test
from natsr.train import train
from natsr.utils import get_config, initialize_torch


def main():
    config = get_config(CONFIG_FILENAME)

    initialize_torch(config)

    mode: str = config['mode']
    if mode == Mode.TRAIN:
        train(config)
    elif mode == Mode.TEST:
        test(config)
    elif mode == Mode.INFERENCE:
        inference(config)
    else:
        raise NotImplementedError(f'[-] not supported mode : {mode}')


main()
