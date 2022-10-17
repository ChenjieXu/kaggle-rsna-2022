#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-04 20:40:34
LastEditors  : ChenjieXu
LastEditTime : 2022-10-17 21:50:35
FilePath     : /cervical/scripts/utils.py
Description  : 
'''
import datetime
import os
import os.path as osp
import glob

import torch
import yaml
from addict import Dict
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def get_exp_name():

    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_config(path='config.yml'):
    if not os.path.exists(path):
        FileNotFoundError
    with open(path, 'r') as f:
        cfg = Dict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg


def save_config(cfg):
    dir_path = os.path.join(cfg.Global.output_dir, cfg.exp_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = osp.join(dir_path, 'config.yml')
    with open(file_path, 'w') as f:
        yaml.dump(cfg.to_dict(), f)
    return cfg



def load_dicom(dir_path):
    """
    This supports loading both regular and compressed JPEG images. 
    See the first sell with `pip install` commands for the necessary dependencies
    """
    ls_imgs = glob.glob(os.path.join(dir_path, "*.dcm"))
    ls_imgs = sorted(ls_imgs, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    imgs = []
    for p_img in ls_imgs:
        dicom = pydicom.dcmread(p_img)
        img = apply_voi_lut(dicom.pixel_array, dicom)
        imgs.append(img.tolist())
    image = torch.tensor(imgs, dtype=torch.float32)
    return image