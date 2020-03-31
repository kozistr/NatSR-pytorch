from natsr import CONFIG_FILENAME
from natsr.utils import get_config


def main():
    config = get_config(CONFIG_FILENAME)
    print(config)


main()
