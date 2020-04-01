from natsr import DataType
from natsr.dataloader import build_data_loader


def train(config):
    train_data_loader = build_data_loader(config, data_type=DataType.TRAIN)
    valid_data_loader = build_data_loader(config, data_type=DataType.VALID)
