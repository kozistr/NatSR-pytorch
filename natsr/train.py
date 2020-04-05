from torch import Tensor
from torch.autograd import Variable

from natsr import ModelType
from natsr.dataloader import build_loader
from natsr.losses import (build_adversarial_loss, build_classification_loss,
                          build_reconstruction_loss)
from natsr.model import build_model
from natsr.optimizers import build_optimizers
from natsr.schedulers import build_lr_scheduler
from natsr.utils import load_model


def nmd_trainer(config, model_type: str, device: str):
    train_loader, valid_loader = build_loader(config)

    nmd_network = build_model(config, model_type, device)
    start_epochs = load_model(
        config['checkpoint']['nmd_model_path'], nmd_network, device
    )

    nmd_optimizer = build_optimizers(config, model_type, nmd_network)

    cls_loss = build_classification_loss(config['model']['cls_loss_type']).to(
        device
    )

    for epoch in range(
        start_epochs, config['model'][model_type]['epochs'] + 1
    ):
        for lr, hr in train_loader:
            pass


def natsr_trainer(config, model_type: str, device: str):
    train_loader, valid_loader = build_loader(config)

    gen_network, disc_network, nmd_network = build_model(
        config, model_type, device
    )
    start_epochs = load_model(
        config['checkpoint']['gen_model_path'], gen_network, device
    )
    _ = load_model(
        config['checkpoint']['disc_model_path'], disc_network, device
    )
    _ = load_model(config['checkpoint']['nmd_model_path'], nmd_network, device)

    gen_optimizer = build_optimizers(config, model_type, gen_network)
    gen_lr_scheduler = build_lr_scheduler(config, model_type, gen_optimizer)

    disc_optimizer = build_optimizers(config, model_type, disc_network)
    disc_lr_scheduler = build_lr_scheduler(config, model_type, disc_optimizer)

    adv_loss = build_adversarial_loss(config['model']['adv_loss_type']).to(
        device
    )
    rec_loss = build_reconstruction_loss(config['model']['rec_loss_type']).to(
        device
    )
    cls_loss = build_classification_loss(config['model']['cls_loss_type']).to(
        device
    )

    for epoch in range(
        start_epochs, config['model'][model_type]['epochs'] + 1
    ):
        for lr, hr in train_loader:
            real = Variable(
                Tensor(lr.size(0), 1).fill_(1.0), requires_grad=False
            )
            fake = Variable(
                Tensor(lr.size(0), 1).fill_(0.0), requires_grad=False
            )

            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            sr = gen_network(lr)
            d_real = disc_network(hr)
            d_fake = disc_network(sr)

            d_loss = adv_loss(d_real, real) + adv_loss(d_fake, fake)
            g_loss = adv_loss(d_fake, real)

            loss = (
                config['model'][ModelType.NATSR]['recon_weight'] * rec_loss(sr, hr)
                + config['model'][ModelType.NATSR]['natural_weight'] * cls_loss
                + config['model'][ModelType.NATSR]['generate_weight']
                * adv_loss
            )


def train(config):
    model_type: str = config['model']['model_type']
    device: str = config['aux']['device']

    if model_type == ModelType.NATSR:
        natsr_trainer(config, model_type, device)
    if model_type == ModelType.NMD:
        nmd_trainer(config, model_type, device)
    raise NotImplementedError(f'[-] not supported modeL_type : {model_type}')
