#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-04 21:02:48
LastEditors  : ChenjieXu
LastEditTime : 2022-10-13 14:26:20
FilePath     : /cervical/scripts/models.py
Description  : 
'''
import torch
import pytorch_lightning as pl

from scripts.factory import build_loss, build_backbone, build_activation, build_optimizer, build_scheduler


class RSNAModel(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(self.cfg)
        self.activation = build_activation(self.cfg)
        self.loss = build_loss(self.cfg)
        self.save_hyperparameters()

    def forward(self, x):
        x = self.backbone(x)
        if self.activation:
            x = self.activation(x)
        return x

    def predict(self, x):
        return torch.sigmoid(self.forward(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        self.log("train_loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        self.log("val_loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = build_optimizer(self, self.cfg)
        lr_scheduler = build_scheduler(optimizer, self.trainer.max_epochs,
                                       self.cfg)

        return [optimizer], [lr_scheduler]