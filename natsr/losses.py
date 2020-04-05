import torch
import torch.nn as nn
from torch.nn.functional import relu

from natsr import AdvLossType, ClsLossType, RecLossType


def generator_loss(adv_loss_type: str, use_ra: bool, real, fake):
    real_loss = 0.0

    if use_ra and adv_loss_type == AdvLossType.WGAN:
        use_ra = False

    if use_ra:
        real_logit = real - torch.mean(fake)
        fake_logit = fake - torch.mean(real)

        if (
            adv_loss_type == AdvLossType.GAN
            or adv_loss_type == AdvLossType.DRAGAN
        ):
            real_loss = nn.BCELoss(reduction='mean')(
                real_logit, torch.zeros_like(real_logit)
            )
            fake_loss = nn.BCELoss(reduction='mean')(
                fake_logit, torch.ones_like(fake_logit)
            )
        elif adv_loss_type == AdvLossType.LSGAN:
            real_loss = torch.mean(((real_logit + 1.0) ** 2))
            fake_loss = torch.mean(((fake_logit - 1.0) ** 2))
        elif adv_loss_type == AdvLossType.HINGE:
            real_loss = torch.mean(relu(1.0 + real))
            fake_loss = torch.mean(relu(1.0 - fake))
        else:
            raise NotImplementedError(
                f'[-] not supported adv_loss_type : {adv_loss_type}'
            )
    else:
        if (
            adv_loss_type == AdvLossType.GAN
            or adv_loss_type == AdvLossType.DRAGAN
        ):
            fake_loss = -nn.BCELoss(reduction='mean')(
                fake, torch.ones_like(fake)
            )
        elif (
            adv_loss_type == AdvLossType.WGANGP
            or adv_loss_type == AdvLossType.WGANLP
        ):
            fake_loss = -torch.mean(fake)
        elif adv_loss_type == AdvLossType.LSGAN:
            fake_loss = torch.mean((fake - 1.0) ** 2)
        elif adv_loss_type == AdvLossType.HINGE:
            fake_loss = -torch.mean(fake)
        else:
            raise NotImplementedError(
                f'[-] not supported adv_loss_type : {adv_loss_type}'
            )

    loss = real_loss + fake_loss

    return loss


def discriminator_loss(adv_loss_type: str, use_ra: bool, real, fake):
    if use_ra and adv_loss_type == AdvLossType.WGAN:
        use_ra = False

    if use_ra:
        real_logit = real - torch.mean(real)
        fake_logit = fake - torch.mean(fake)

        if (
            adv_loss_type == AdvLossType.GAN
            or adv_loss_type == AdvLossType.DRAGAN
        ):
            real_loss = nn.BCELoss(reduction='mean')(
                real_logit, torch.ones_like(real_logit)
            )
            fake_loss = nn.BCELoss(reduction='mean')(
                fake_logit, torch.zeros_like(fake_logit)
            )
        elif adv_loss_type == AdvLossType.LSGAN:
            real_loss = torch.mean(((real_logit - 1.0) ** 2))
            fake_loss = torch.mean(((fake_logit + 1.0) ** 2))
        elif adv_loss_type == AdvLossType.HINGE:
            real_loss = torch.mean(relu(1.0 - real))
            fake_loss = torch.mean(relu(1.0 + fake))
        else:
            raise NotImplementedError(
                f'[-] not supported adv_loss_type : {adv_loss_type}'
            )
    else:
        if (
            adv_loss_type == AdvLossType.GAN
            or adv_loss_type == AdvLossType.DRAGAN
        ):
            real_loss = nn.BCELoss(reduction='mean')(
                real, torch.ones_like(real)
            )
            fake_loss = nn.BCELoss(reduction='mean')(
                fake, torch.zeros_like(fake)
            )
        elif (
            adv_loss_type == AdvLossType.WGANGP
            or adv_loss_type == AdvLossType.WGANLP
        ):
            real_loss = -torch.mean(real)
            fake_loss = torch.mean(fake)
        elif adv_loss_type == AdvLossType.LSGAN:
            real_loss = torch.mean(((real - 1.0) ** 2))
            fake_loss = torch.mean(fake ** 2)
        elif adv_loss_type == AdvLossType.HINGE:
            real_loss = torch.mean(relu(1.0 - real))
            fake_loss = torch.mean(relu(1.0 + fake))
        else:
            raise NotImplementedError(
                f'[-] not supported adv_loss_type : {adv_loss_type}'
            )

    loss = real_loss + fake_loss

    return loss


def build_classification_loss(cls_loss_type: str):
    if cls_loss_type == ClsLossType.BCE:
        return nn.BCELoss()
    if cls_loss_type == ClsLossType.CCE:
        return nn.CrossEntropyLoss()
    raise NotImplementedError(
        f'[-] not supported cls_loss_type : {cls_loss_type}'
    )


def build_reconstruction_loss(rec_loss_type: str):
    if rec_loss_type == RecLossType.L1:
        return nn.L1Loss()
    if rec_loss_type == RecLossType.L2:
        return nn.MSELoss()
    raise NotImplementedError(
        f'[-] not supported adv_loss_type : {rec_loss_type}'
    )


def build_perceptual_loss(_):
    raise NotImplementedError('[-] not implemented yet')
