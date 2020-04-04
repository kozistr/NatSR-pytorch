import torch.nn as nn

from natsr import AdvLossType, ClsLossType, RecLossType


def build_adversarial_loss(adv_loss_type: str):
    if adv_loss_type == AdvLossType.GAN:
        return nn.BCELoss()
    raise NotImplementedError(
        f'[-] not supported adv_loss_type : {adv_loss_type}'
    )


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
