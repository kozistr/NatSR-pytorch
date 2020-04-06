import numpy as np
import torch

from natsr import ModelType
from natsr.dataloader import build_loader
from natsr.losses import (
    build_reconstruction_loss,
    discriminator_loss,
    generator_loss,
    natural_loss,
)
from natsr.metrics import ssim
from natsr.model import build_model
from natsr.optimizers import build_optimizers
from natsr.schedulers import build_lr_scheduler
from natsr.utils import (
    build_summary_writer,
    load_models,
    log_summary,
    tensor_to_numpy,
)


def nmd_trainer(config, model_type: str, device: str, summary):
    train_loader, valid_loader = build_loader(config)

    nmd_network = build_model(config, model_type, device)
    start_epochs, _ = load_models(
        config['checkpoint']['nmd_model_path'], device, None, None, nmd_network
    )

    nmd_optimizer = build_optimizers(config, model_type, nmd_network)

    nmd_network.train()

    for epoch in range(
            start_epochs, config['model'][model_type]['epochs'] + 1
    ):
        for lr, hr in train_loader:
            pass


def natsr_trainer(config, model_type: str, device: str, summary):
    train_loader, valid_loader = build_loader(config)

    gen_network, disc_network, nmd_network = build_model(
        config, model_type, device
    )
    start_epochs, start_ssim = load_models(
        config, device, gen_network, disc_network, nmd_network
    )
    end_epochs: int = config['model'][model_type]['epochs'] + 1

    gen_network.to(device)
    disc_network.to(device)
    nmd_network.to(device)

    gen_optimizer = build_optimizers(config, model_type, gen_network)
    gen_lr_scheduler = build_lr_scheduler(config, model_type, gen_optimizer)

    disc_optimizer = build_optimizers(config, model_type, disc_network)
    disc_lr_scheduler = build_lr_scheduler(config, model_type, disc_optimizer)

    recon_loss = build_reconstruction_loss(
        config['model']['rec_loss_type']
    ).to(device)

    gen_network.train()
    disc_network.train()

    best_ssim: float = start_ssim
    global_step: int = start_epochs * len(
        train_loader
    ) // train_loader.batch_size
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
            nat_loss = natural_loss(sr).to(device)

            loss = (
                    config['model'][ModelType.NATSR]['recon_weight']
                    * recon_loss(sr, hr)
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
                    and global_step % config['log']['logging_step'] == 0
            ):
                gen_network.eval()
                disc_network.eval()

                logs = {
                    'loss/total_loss': loss.item(),
                    'loss/adv_loss': g_loss.item(),
                    'loss/rec_loss': rec_loss.item(),
                    'loss/nat_loss': nat_loss.item(),
                    'aux/g_lr': gen_lr_scheduler.get_lr(),
                    'aux/d_lr': disc_lr_scheduler.get_lr(),
                    'sr': torch.clamp(0.0, 1.0, sr),
                    'hr': torch.clamp(0.0, 1.0, hr),
                }
                log_summary(summary, logs, global_step)

                with torch.no_grad():
                    curr_ssim = np.mean(
                        [
                            ssim(
                                tensor_to_numpy(torch.clamp(0.0, 1.0, _sr)),
                                tensor_to_numpy(torch.clamp(0.0, 1.0, _hr)),
                            )
                            for _sr, _hr in zip(gen_network(val_lr), val_hr)
                            for val_lr, val_hr in valid_loader
                        ],
                        dtype=np.float32,
                    )

                if curr_ssim > best_ssim:
                    print(
                        f'[{epoch}/{end_epochs}] [{global_step} steps] '
                        f'SSIM improved from {curr_ssim} to {best_ssim}'
                    )
                    best_ssim = curr_ssim

                gen_network.train()
                disc_network.train()

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

    summary = build_summary_writer(config)

    if model_type == ModelType.NATSR:
        natsr_trainer(config, model_type, device, summary)
    if model_type == ModelType.NMD:
        nmd_trainer(config, model_type, device, summary)
    raise NotImplementedError(f'[-] not supported modeL_type : {model_type}')
