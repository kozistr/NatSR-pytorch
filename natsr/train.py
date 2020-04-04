from natsr import ModelType
from natsr.dataloader import build_loader
from natsr.losses import (
    build_adversarial_loss,
    build_classification_loss,
    build_reconstruction_loss,
)
from natsr.model import build_model
from natsr.optimizers import build_optimizers
from natsr.schedulers import build_lr_scheduler


def nmd_trainer(config, model_type: str):
    train_loader, valid_loader = build_loader(config)

    nmd_network = build_model(config, model_type)
    nmd_optimizer = build_optimizers(config, model_type, nmd_network)
    cls_loss = build_classification_loss(config['model']['cls_loss_type'])


def natsr_trainer(config, model_type: str):
    train_loader, valid_loader = build_loader(config)

    gen_network, disc_network = build_model(config, model_type)

    gen_optimizer = build_lr_scheduler(
        config, build_optimizers(config, model_type, gen_network)
    )
    disc_optimizer = build_lr_scheduler(
        config, build_optimizers(config, model_type, disc_network)
    )

    adv_loss = build_adversarial_loss(config['model']['adv_loss_type'])
    rec_loss = build_reconstruction_loss(config['model']['rec_loss_type'])


def train(config):
    model_type: str = config['model']['model_type']

    if model_type == ModelType.NATSR:
        natsr_trainer(config, model_type)
    elif model_type == ModelType.NMD:
        nmd_trainer(config, model_type)
    raise NotImplementedError(f'[-] not supported modeL_type : {model_type}')
