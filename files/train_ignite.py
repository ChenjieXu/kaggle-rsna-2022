#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-05 17:57:30
LastEditors  : ChenjieXu
LastEditTime : 2022-10-12 08:10:32
FilePath     : /cervical/files/train_ignite.py
Description  : 
'''

import warnings

import torch
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from monai.transforms import (Compose, EnsureChannelFirst, Resize,
                              ScaleIntensity)
from monai.utils.misc import set_determinism

from dataloader import build_dataloader
from scripts.models import (build_loss, build_lr_scheduler_ignite, build_backbone,
                      build_optimizer, set_evaluator, set_trainer)
from scripts.utils import RSNA_Metric, get_exp_name, load_config, save_config

warnings.filterwarnings("ignore")

def run():

    set_determinism()
    cfg = load_config('exp_config.yml')
    cfg.exp_name = get_exp_name()
    
    save_config(cfg)

    
    # TODO:增加数据增强
    train_transforms = Compose(
        [
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((112, 112, 80))])
    val_transforms = Compose(
        [
        ScaleIntensity(),
        EnsureChannelFirst(),
        Resize((112, 112, 80))])

    # 准备数据集和数据加载器
    train_loader, val_loader = build_dataloader(train_transforms, val_transforms, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_backbone(cfg)
    loss = build_loss(cfg)
    optimizer = build_optimizer(net, cfg)
    lr_scheduler = build_lr_scheduler_ignite(optimizer, cfg, train_loader)


    # 创建Trainer
    trainer = create_supervised_trainer(net, optimizer, loss, device, False)

    # TODO: 添加logging
    set_trainer(trainer, net, optimizer, lr_scheduler, cfg)

    # 创建Evaluator
    cfg.metric_name = "Weighted Loss"
    val_metrics = {cfg.metric_name: RSNA_Metric(device=device)}
    evaluator = create_supervised_evaluator(net, val_metrics, device, True)

    set_evaluator(trainer, evaluator, cfg)

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.Global.validation_every_n_epochs))
    def run_validation(engine):
        evaluator.run(val_loader)

    # 开始训练
    state = trainer.run(train_loader, cfg.Global.epoch_num)


if __name__ == '__main__':
    run()
