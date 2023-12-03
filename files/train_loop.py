#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-12 07:10:36
LastEditors  : ChenjieXu
LastEditTime : 2022-10-12 07:46:28
FilePath     : /cervical/train_loop.py
Description  : 
'''

import gc
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.notebook import tqdm
from scripts.factory import build_loss, build_backbone, build_optimizer, build_scheduler
def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

def save_model(name, model):
    torch.save(model.state_dict(), f'{name}.tph')
def load_model(model, name, path='.'):
    data = torch.load(os.path.join(path, f'{name}.tph'))
    model.load_state_dict(data)
    return model

def train_loop(model, dataloader, logger, name, cfg, device):
    torch.manual_seed(42)

    dl_train = dataloader.train_dataloader()
    ds_eval = dataloader.val_dataloader()

    optim = build_optimizer(model, cfg)
    scheduler = build_scheduler(optim, cfg.Global.epoch_num, cfg)

    model.train()
    scaler = GradScaler()
    with tqdm(dl_train, desc='Train', miniters=10) as progress:
        for batch_idx, batch in enumerate(progress):
            batch = [x.to(device) for x in batch]
            if ds_eval is not None and batch_idx % cfg.Global.validation_every_n_epochs == 0:
                model.eval()
                loss = model.training_step(batch, batch_idx)
                model.train()
                print(({'eval_loss': loss}))
                if batch_idx > 0:  # don't save untrained model
                    save_model(name, model)

            if batch_idx >= cfg.Global.epoch_num:
                break

            optim.zero_grad()
            # Using mixed precision training
            with autocast():
                loss = model.training_step(batch, batch_idx)

                if np.isinf(loss.item()) or np.isnan(loss.item()):
                    print(f'Bad loss, skipping the batch {batch_idx}')
                    del loss
                    gc_collect()
                    continue

            # scaler is needed to prevent "gradient underflow"
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            progress.set_description(f'Train loss: {loss.item() :.02f}')
            print(str({
                'loss': (loss.item()),
                'lr': scheduler.get_last_lr()[0]
            }))

    save_model(name, model)
    return model
