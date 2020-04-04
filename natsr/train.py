from natsr.dataloader import build_loader
from natsr.model import build_model


def train(config):
    train_loader, valid_loader = build_loader(config)
    gen_network, nmd_network, disc_network = build_model(config)
