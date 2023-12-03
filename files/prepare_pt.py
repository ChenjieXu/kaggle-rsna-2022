#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-07 19:06:33
LastEditors: 
LastEditTime: 2023-12-02 18:58:32
FilePath     : /cervical/prepare_pt.py
Description  : 
'''

# !pip install -qU "python-gdcm" pydicom pylibjpeg

import cv2
import os
import glob
import numpy as np
import torch
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers import apply_voi_lut
from skimage import exposure
from tqdm.auto import tqdm
from joblib import Parallel, delayed

def convert_volume(dir_path: str, out_dir: str = "test_volumes", size=(256, 256, 224)):
    # 获取所有.dcm文件的路径
    ls_imgs = glob.glob(os.path.join(dir_path, "*.dcm"))
    # 按文件名排序以确保顺序一致性
    ls_imgs = sorted(ls_imgs, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    imgs = []
    # 遍历每个.dcm文件
    for p_img in ls_imgs:
        try:
            # 读取.dcm文件
            dicom = pydicom.dcmread(p_img)
            # 应用VOI LUT（窗位窗宽调整）
            img = apply_voi_lut(dicom.pixel_array, dicom)
            # 将图像调整为指定大小
            img = cv2.resize(img, size[:2], interpolation=cv2.INTER_LINEAR)
            imgs.append(img.tolist())
        except Exception as e:
            # 处理异常情况，如文件格式错误
            print(f"Error processing file {p_img}: {e}")

    # 将图像列表转换为PyTorch张量
    vol = torch.tensor(imgs, dtype=torch.float32)
    # 构建输出路径
    path_pt = os.path.join(out_dir, f"{os.path.basename(dir_path)}.pt")
    # 保存PyTorch张量到文件
    torch.save(vol, path_pt)

PATH_DATASET = ""
TRAIN_IMAGES_PATH = os.path.join(PATH_DATASET, "train_images")
OUTPUT_DIR = os.path.join(PATH_DATASET, "train_pt")

# 获取所有训练图像目录的列表
ls_dirs = [os.path.join(TRAIN_IMAGES_PATH, p) for p in os.listdir(TRAIN_IMAGES_PATH) if os.path.isdir(os.path.join(TRAIN_IMAGES_PATH, p))]
print(f"Total volumes: {len(ls_dirs)}")
# 移除已存在的输出文件
ls_dirs_existing = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(OUTPUT_DIR, "*.pt"))]
ls_dirs = [p for p in ls_dirs if p not in ls_dirs_existing]
print(f"Volumes to process: {len(ls_dirs)}")
# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 构建完整的目录路径
ls_dirs = [os.path.join(PATH_DATASET, "train_images", p) for p in ls_dirs]

# 使用joblib的parallel_backend进行并行处理
with Parallel(n_jobs=-1, backend="loky") as parallel:
    # 对每个目录并行处理
    parallel(delayed(convert_volume)(p_dir, out_dir=OUTPUT_DIR) for p_dir in tqdm(ls_dirs))
