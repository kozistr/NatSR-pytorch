import numpy as np
import torch

from natsr import ModelType
from natsr.dataloader import build_loader, get_nmd_data
from natsr.losses import (
    build_classification_loss,
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
    save_model,
    tensor_to_numpy,
)


def nmd_trainer(config, model_type: str, device: str, summary):
    train_loader, valid_loader = build_loader(config, override_batch_size=100)

    nmd_network = build_model(config, model_type, device)
    start_epochs, _, alpha, sigma = load_models(
        config['checkpoint']['nmd_model_path'], device, None, None, nmd_network
    )

    nmd_optimizer = build_optimizers(config, model_type, nmd_network)

    cls_loss = build_classification_loss(
        config['model']['cls_loss_type'], device
    )

    nmd_network.train()

    global_step: int = start_epochs * len(
        train_loader
    ) // train_loader.batch_size
    for epoch in range(
        start_epochs, config['model'][model_type]['epochs'] + 1
    ):
        for _, lr_img in train_loader:
            train_img = get_nmd_data(lr_img, alpha, sigma)
            train_label = torch.cat(
                [
                    torch.zeros((lr_img.size(0) // 2, 1, 1, 1)),
                    torch.ones((lr_img.size(0) // 2, 1, 1, 1)),
                ]
            ).to(device)

            out = nmd_network(train_img.to(device))

            nmd_optimizer.zero_grad()
            loss = cls_loss(out, train_label)
            loss.backward()
            nmd_optimizer.step()

            if (
                global_step
                and global_step % config['log']['logging_step'] == 0
            ):
                nmd_network.eval()

                with torch.no_grad():
                    for _, val_lr_img in valid_loader:
                        valid_img = get_nmd_data(val_lr_img, alpha, sigma)
                        valid_label = torch.cat(
                            [
                                torch.zeros(
                                    (val_lr_img.size(0) // 2, 1, 1, 1)
                                ),
                                torch.ones((val_lr_img.size(0) // 2, 1, 1, 1)),
                            ]
                        ).to(device)

                        logs = {
                            'loss/cls_loss': loss.item(),
                            'metric/noisy_acc': 0,
                            'metric/blurry_acc': 0,
                            'aux/alpha': alpha,
                            'aux/sigma': sigma,
                        }
                        log_summary(summary, logs, global_step)

                nmd_network.train()

            if alpha >= 0.8 and sigma <= 0.0044:
                print(
                    f'[+] training NMD is done! '
                    f'alpha : {alpha} sigma : {sigma}'
                )

                save_model(
                    config['log']['checkpoint']['nmd_model_path'],
                    nmd_network,
                    epoch,
                    0.0,
                    alpha,
                    sigma,
                )
                break

            global_step += 1


def frsr_trainer(config, model_type: str, device: str, summary):
    train_loader, valid_loader = build_loader(config)

    gen_network, disc_network, nmd_network = build_model(
        config, model_type, device
    )
    start_epochs, start_ssim, _, _ = load_models(
        config, device, gen_network, disc_network, nmd_network
    )
    end_epochs: int = config['model'][model_type]['epochs'] + 1

    gen_network = gen_network.to(device)
    disc_network = disc_network.to(device)
    nmd_network = nmd_network.to(device)

    gen_optimizer = build_optimizers(config, model_type, gen_network)
    gen_lr_scheduler = build_lr_scheduler(config, model_type, gen_optimizer)

    disc_optimizer = build_optimizers(config, model_type, disc_network)
    disc_lr_scheduler = build_lr_scheduler(config, model_type, disc_optimizer)

    recon_loss = build_reconstruction_loss(
        config['model']['rec_loss_type'], device
    )

    gen_network.train()
    disc_network.train()

    best_ssim: float = start_ssim
    global_step: int = start_epochs * len(
        train_loader
    ) // train_loader.batch_size
    for epoch in range(start_epochs, end_epochs):
        for lr, hr in train_loader:
            gen_optimizer.zero_grad()

            sr = gen_network(lr.to(device))
            d_real = disc_network(hr.to(device))
            d_fake = disc_network(sr)
            nat = nmd_network(sr)

            g_loss = generator_loss(
                config['model']['adv_loss_type'],
                bool(config['model']['use_ra']),
                d_real,
                d_fake,
            ).to(device)
            nat_loss = natural_loss(nat).to(device)
            rec_loss = recon_loss(sr, hr.to(device))

            loss = (
                config['model'][ModelType.FRSR]['loss']['recon_weight']
                * rec_loss
                + config['model'][ModelType.FRSR]['loss']['natural_weight']
                * nat_loss
                + config['model'][ModelType.FRSR]['loss']['generate_weight']
                * g_loss
            )
            loss.backward(retain_graph=True)
            gen_optimizer.step()

            disc_optimizer.zero_grad()

            d_loss = discriminator_loss(
                config['model']['adv_loss_type'],
                bool(config['model']['use_ra']),
                d_real,
                d_fake,
            ).to(device)

            d_loss.backward(retain_graph=True)
            disc_optimizer.step()

            if global_step % config['log']['logging_step'] == 0:
                gen_network.eval()
                disc_network.eval()

                with torch.no_grad():
                    curr_ssim = np.mean(
                        [
                            ssim(
                                tensor_to_numpy(
                                    torch.clamp(0.0, 1.0, _sr.to(device))
                                ),
                                tensor_to_numpy(
                                    torch.clamp(0.0, 1.0, _hr.to(device))
                                ),
                            )
                            for val_lr, val_hr in valid_loader
                            for _sr, _hr in zip(
                                gen_network(val_lr.to(device)), val_hr
                            )
                        ],
                        dtype=np.float32,
                    )

                if curr_ssim > best_ssim:
                    print(
                        f'[{epoch}/{end_epochs}] [{global_step} steps] '
                        f'SSIM improved from {curr_ssim} to {best_ssim}'
                    )
                    best_ssim = curr_ssim

                    save_model(
                        config['log']['checkpoint']['gen_model_path'],
                        gen_network,
                        epoch,
                        best_ssim,
                    )
                    save_model(
                        config['log']['checkpoint']['disc_model_path'],
                        disc_network,
                        epoch,
                        best_ssim,
                    )
                    save_model(
                        config['log']['checkpoint']['nmd_model_path'],
                        nmd_network,
                        epoch,
                        best_ssim,
                    )

                logs = {
                    'loss/total_loss': loss.item(),
                    'loss/adv_loss': g_loss.item(),
                    'loss/rec_loss': rec_loss.item(),
                    'loss/nat_loss': nat_loss.item(),
                    'metric/ssim': best_ssim,
                    'aux/g_lr': gen_lr_scheduler.get_lr(),
                    'aux/d_lr': disc_lr_scheduler.get_lr(),
                    'sr': torch.clamp(0.0, 1.0, sr),
                    'hr': torch.clamp(0.0, 1.0, hr),
                }
                log_summary(summary, logs, global_step)

                gen_network.train()
                disc_network.train()

            if (
                global_step
                and global_step
                % config['model'][ModelType.FRSR]['lr_decay_steps']
                == 0
            ):
                gen_lr_scheduler.step()
                disc_lr_scheduler.step()

            global_step += 1


def train(config):
    model_type: str = config['model']['model_type']
    device: str = config['aux']['device']

    summary = build_summary_writer(config)

    if model_type == ModelType.FRSR:
        frsr_trainer(config, model_type, device, summary)
    if model_type == ModelType.NMD:
        nmd_trainer(config, model_type, device, summary)
    raise NotImplementedError(f'[-] not supported modeL_type : {model_type}')
