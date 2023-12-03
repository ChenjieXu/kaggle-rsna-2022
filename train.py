#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-06 23:06:53
LastEditors  : ChenjieXu
LastEditTime : 2022-10-23 10:02:49
FilePath     : /projects/cervical/train.py
Description  : 
'''

import logging
import warnings
import argparse

import pytorch_lightning as pl

from scripts.dataloaders import RSNADataModule
from scripts.factory import build_callbacks, build_loggers
from scripts.models import RSNAModel
from scripts.utils import load_config, save_config

warnings.filterwarnings("ignore")

logger = logging.getLogger("pytorch_lightning")
logger.addHandler(logging.FileHandler("core.log"))

def parse_args():  
    """添加参数信息"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, help='config path')
    args = parser.parse_args()
    return args


def run(args):
    # 准备参数
    cfg = load_config(args.config)
    save_config(cfg)

    cfg.devices = [2]
    cfg.Optimizer.lr *= len(cfg.devices) if isinstance(
        cfg.devices, list) else int(cfg.devices)
    
    # 准备数据集
    rsna_datamodule = RSNADataModule(cfg)
    rsna_datamodule.setup()
    cfg.steps_per_epoch = len(rsna_datamodule.train_dataloader())

    loggers = build_loggers(cfg)
    callbacks = build_callbacks(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.Global.epoch_num,
        devices=cfg.devices,
        auto_select_gpus=True,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.Global.validation_every_n_epochs,
        accelerator="gpu",
        precision=16,
        strategy="ddp_find_unused_parameters_false",
        benchmark=True,  # 模型固定输入大小时加速
        #  profiler="simple",
        # auto_scale_batch_size="binsearch",
        # limit_train_batches=10, limit_val_batches=5,
        # overfit_batches=10,
    )

    # 模型初始化，加载权重来finetune
    if cfg.Global.weights is not None:
        model = RSNAModel.load_from_checkpoint(cfg.Global.weights)
        print(f"weights from: {cfg.Global.weights} are loaded.")
    else:
        model = RSNAModel(cfg)
    # 开始训练
    trainer.fit(
        model,
        datamodule=rsna_datamodule,
    )


if __name__ == '__main__':
    args = parse_args()
    run(args)
