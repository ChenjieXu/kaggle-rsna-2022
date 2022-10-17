#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author       : ChenjieXu
Date         : 2022-10-11 20:08:37
LastEditors  : ChenjieXu
LastEditTime : 2022-10-13 21:17:10
FilePath     : /cervical/scripts/factory.py
Description  : 
"""
import logging
import os

import monai
import numpy as np
import torch
from monai.transforms import (Compose, EnsureChannelFirst, EnsureType, OneOf,
                              RandAffine, RandCoarseDropout, RandFlip,
                              RandGridDistortion, RandScaleIntensity,RandAdjustContrast,
                              RandShiftIntensity, Resize, ScaleIntensity)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from scripts.losses import CustomLoss, CustomWithLogitsLoss


def build_activation(cfg):
    if cfg.Model.activation == "Sigmoid":
        activation = torch.nn.Sigmoid()
    elif cfg.Model.activation == "None":
        activation = None
    else:
        raise NotImplementedError
    return activation


def build_backbone(cfg):
    if cfg.Model.architecture == "DenseNet":
        backbone = monai.networks.nets.DenseNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=cfg.Global.num_classes,
        )
    elif cfg.Model.architecture == "DenseNet169":
        backbone = monai.networks.nets.DenseNet169(
            spatial_dims=3,
            in_channels=1,
            out_channels=cfg.Global.num_classes,
        )
    elif cfg.Model.architecture == "efficientnet-b2":
        backbone = monai.networks.nets.EfficientNetBN(
            "efficientnet-b2",
            spatial_dims=3,
            in_channels=1,
            num_classes=cfg.Global.num_classes,
        )
    return backbone


def build_loss(cfg):
    if cfg.Loss.name == "CustomWithLogitsLoss":
        loss = CustomWithLogitsLoss()
    elif cfg.Loss.name == "CustomLoss":
        loss = CustomLoss()
    elif cfg.Loss.name == "BCEWithLogitsLoss":
        loss = torch.nn.BCEWithLogitsLoss()
    else:
        NotImplementedError
    return loss


def build_optimizer(net, cfg):

    optimizer = torch.optim.Adam(net.parameters(), cfg.Optimizer.lr,
                                 (cfg.Optimizer.beta1, cfg.Optimizer.beta2))

    return optimizer


def build_scheduler(optimizer, T_max, cfg):
    if cfg.Lr_scheduler.name == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False)
    elif cfg.Lr_scheduler.name == "warmup_restart":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max)
    elif cfg.Lr_scheduler.name == "OneCycleLR":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, cfg.Optimizer.lr, epochs=T_max, steps_per_epoch=cfg.steps_per_epoch,)
    return lr_scheduler


def build_callbacks(cfg):

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.Global.output_dir, cfg.exp_name, "models"),
        save_top_k=5,
        monitor="val_loss",
        filename="rsna-{epoch:02d}-{val_loss:.3f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    return [checkpoint_callback, lr_monitor]


def build_loggers(cfg):
    logger_dir = os.path.join(cfg.Global.output_dir, cfg.exp_name, "logs")
    os.makedirs(logger_dir,exist_ok=True)
    csv_logger = CSVLogger(save_dir=logger_dir)
    wandb_logger = WandbLogger(project="CERVICAL")
    return [csv_logger, wandb_logger]


def build_transforms(cfg):
    train_transform = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize(cfg.Model.input_size),
        RandAdjustContrast(prob=0.1, gamma=(0.5, 4.5)),
        # RandSpatialCrop(
        #     roi_size=cfg.img_size,
        #     random_size=False,
        # ),
        RandFlip(prob=0.1, spatial_axis=[0]),
        RandFlip(prob=0.1, spatial_axis=[1]),
        # RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[2]),
        RandAffine(
            prob=0.1,
            rotate_range=np.pi / 12,
            translate_range=(cfg.Model.input_size[0] * 0.0625,
                             cfg.Model.input_size[0] * 0.0625),
            scale_range=(0.1, 0.1),
            mode="nearest",
            padding_mode="zeros",
        ),
        RandScaleIntensity(factors=(-0.2, 0.2), prob=0.1),
        RandShiftIntensity(offsets=(-0.2, 0.2), prob=0.1),
        EnsureType(dtype=torch.float32),
    ])

    val_transform = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize(cfg.Model.input_size),
        EnsureType(dtype=torch.float32),
    ])

    return train_transform, val_transform
