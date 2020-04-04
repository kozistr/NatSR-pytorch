from natsr.dataloader import build_loader
from natsr.losses import build_adversarial_loss, build_reconstruction_loss
from natsr.model import build_model
from natsr.optimizers import build_optimizers


def train(config):
    train_loader, valid_loader = build_loader(config)

    gen_network, nmd_network, disc_network = build_model(config)

    gen_optimizer = build_optimizers(config, gen_network)
    nmd_optimizer = build_optimizers(config, nmd_network)
    disc_optimizer = build_optimizers(config, disc_network)

    adv_loss = build_adversarial_loss(config)
    rec_loss = build_reconstruction_loss(config)
