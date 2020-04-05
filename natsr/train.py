from natsr import ModelType
from natsr.dataloader import build_loader
from natsr.losses import (
    build_classification_loss, build_reconstruction_loss, discriminator_loss,
    generator_loss)
from natsr.model import build_model
from natsr.optimizers import build_optimizers
from natsr.schedulers import build_lr_scheduler
from natsr.utils import load_models


def nmd_trainer(config, model_type: str, device: str):
    train_loader, valid_loader = build_loader(config)

    nmd_network = build_model(config, model_type, device)
    start_epochs = load_models(
        config['checkpoint']['nmd_model_path'], device, None, None, nmd_network
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

    start_epochs: int = load_models(
        config, device, gen_network, disc_network, nmd_network
    )
    end_epochs: int = config['model'][model_type]['epochs'] + 1

    gen_optimizer = build_optimizers(config, model_type, gen_network)
    gen_lr_scheduler = build_lr_scheduler(config, model_type, gen_optimizer)

    disc_optimizer = build_optimizers(config, model_type, disc_network)
    disc_lr_scheduler = build_lr_scheduler(config, model_type, disc_optimizer)

    recon_loss = build_reconstruction_loss(
        config['model']['rec_loss_type']
    ).to(device)
    cls_loss = build_classification_loss(config['model']['cls_loss_type']).to(
        device
    )

    global_step: int = 0
    for epoch in range(start_epochs, end_epochs):
        for lr, hr in train_loader:
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            sr = gen_network(lr)
            d_real = disc_network(hr)
            d_fake = disc_network(sr)

            g_loss = generator_loss(
                config['model']['adv_loss_type'],
                bool(config['model']['use_ra']),
                d_real,
                d_fake,
            ).to(device)
            rec_loss = recon_loss(sr, hr)
            nat_loss = cls_loss()

            loss = (
                config['model'][ModelType.NATSR]['recon_weight'] * rec_loss
                + config['model'][ModelType.NATSR]['natural_weight'] * nat_loss
                + config['model'][ModelType.NATSR]['generate_weight'] * g_loss
            )
            loss.backward()
            gen_optimizer.step()

            d_loss = discriminator_loss(
                config['model']['adv_loss_type'],
                bool(config['model']['use_ra']),
                d_real,
                d_fake,
            ).to(device)

            d_loss.backward()
            disc_optimizer.step()

            if (
                global_step
                and global_step % config['aux']['logging_step'] == 0
            ):
                print(
                    f'[Epoch {epoch}/{end_epochs}] '
                    f'[Steps {global_step} '
                    f'[total loss: {loss.item()}] '
                    f'[adv loss: {g_loss.item()}] '
                    f'[rec loss: {rec_loss.item()}] '
                    f'[nat loss: {nat_loss.item()}]'
                )

            if (
                global_step
                and global_step
                % config['model'][ModelType.NATSR]['lr_decay_steps']
                == 0
            ):
                gen_lr_scheduler.step()
                disc_lr_scheduler.step()

            global_step += 1


def train(config):
    model_type: str = config['model']['model_type']
    device: str = config['aux']['device']

    if model_type == ModelType.NATSR:
        natsr_trainer(config, model_type, device)
    if model_type == ModelType.NMD:
        nmd_trainer(config, model_type, device)
    raise NotImplementedError(f'[-] not supported modeL_type : {model_type}')
