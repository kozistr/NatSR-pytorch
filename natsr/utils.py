import yaml


def get_config(filename: str):
    with open(filename, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config
