#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-12 07:03:45
LastEditors  : ChenjieXu
LastEditTime : 2022-10-12 08:10:30
FilePath     : /cervical/files/train_native.py
Description  : 
'''

import os
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from scripts.dataloaders import RSNADataModule
from scripts.models import RSNAModel
from train_loop import train_loop
from scripts.utils import get_exp_name, load_config, save_config

warnings.filterwarnings("ignore")
import logging



def run():
    logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    cfg = load_config('exp_config.yml')
    cfg.exp_name = get_exp_name()
    save_config(cfg)
    device = torch.device("cuda:7")
    # 准备数据集
    rsna_datamodule = RSNADataModule(cfg)
    rsna_datamodule.setup()
    # 模型初始化，并加载权重来finetune
    # TODO: 把这些参数整成pl的格式
    model = RSNAModel(cfg).to(device)
    if cfg.Global.weights is not None:
        model.load_from_checkpoint(cfg.Global.weights, cfg=cfg)
        print(f"weights from: {cfg.Global.weights} are loaded.")

    train_loop(model, rsna_datamodule, logger, '1', cfg,)

if __name__ == '__main__':
    run()
