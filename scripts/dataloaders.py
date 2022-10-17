#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-05 20:53:37
LastEditors  : ChenjieXu
LastEditTime : 2022-10-17 15:21:27
FilePath     : /cervical/scripts/dataloaders.py
Description  : 
'''
import os
import os.path as osp
from typing import Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

from scripts.factory import build_transforms
from scripts.utils import load_dicom



class RSNADataset(Dataset):

    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.transform = transform
        self.images = images
        self.labels = labels
        assert len(self.images) == len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        image = self.images[idx]
        label = self.labels[idx]

        try:
            image = torch.load(image).to(torch.float32)
        except (EOFError, RuntimeError):
            print(f"failed loading: {image}")
        if self.transform is not None:
            image = self.transform(image)
        # in case of predictions, return image name as label
        label = label if label is not None else image
        return (image, label)

    def __len__(self) -> int:
        return len(self.labels)


class RSNADataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_transform, self.val_transform = build_transforms(self.cfg)

    def setup(self, stage: Optional[str] = None):
        # 这样子分割出来的数据集是Subset，没有父类的方法，无法再设置transform
        # # 实例化整个数据集
        # rsna_full = RSNADataset(self.cfg)
        # # 数据集分割
        # totol_num = len(rsna_full)
        # test_size = int(totol_num * self.cfg.Dataset.test_size)
        # self.rsna_train, self.rsna_val = random_split(
        #     rsna_full, [totol_num - test_size, test_size])
        # # 分别设置transform
        # self.rsna_train.super()._update_transform(self.train_transform)
        # self.rsna_val.super()._update_transform(self.val_transform)
        images, labels = prepare_dataset_data(self.cfg)
        train_images, val_images, train_labels, val_labels = train_test_split(
            images,
            labels,
            test_size=self.cfg.Dataset.test_size,
            random_state=42,
        )
        self.rsna_train = RSNADataset(train_images, train_labels,
                                      self.train_transform)
        self.rsna_val = RSNADataset(val_images, val_labels, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.rsna_train,
            batch_size=self.cfg.TrainReader.batch_size,
            shuffle=self.cfg.TrainReader.shuffle,
            #   num_workers=self.cfg.TrainReader.num_workers,
            #   pin_memory=self.cfg.TrainReader.pin_memory,
            drop_last=self.cfg.TrainReader.drop_last)

    def val_dataloader(self):
        return DataLoader(
            self.rsna_val,
            batch_size=self.cfg.ValReader.batch_size,
            shuffle=self.cfg.ValReader.shuffle,
            #   num_workers=self.cfg.ValReader.num_workers,
            #   pin_memory=self.cfg.ValReader.pin_memory,
            drop_last=self.cfg.ValReader.drop_last)


class RSNATestDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        _, self.val_transform = build_transforms(self.cfg)

    def setup(self, stage: Optional[str] = None):
        test_csv = pd.read_csv(
            osp.join(self.cfg.Dataset.dataset_dir, "test.csv"))
        test_csv.drop_duplicates(subset=["StudyInstanceUID"], inplace=True)
        uid_list = test_csv["StudyInstanceUID"].to_list()
        test_images = [
            osp.join(self.cfg.Dataset.dataset_dir, "test_images", uid)
            for uid in uid_list
        ]
        self.rsna_test = RSNADatasetDicom(test_images,  uid_list, self.val_transform)

    def predict_dataloader(self):
        return DataLoader(self.rsna_test,
                          batch_size=self.cfg.TestReader.batch_size,
                          num_workers=os.cpu_count())

class RSNADatasetDicom(Dataset):

    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.transform = transform
        self.images = images
        self.labels = labels
        assert len(self.images) == len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.images[idx]
        label = self.labels[idx]

        image = load_dicom(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)

    def __len__(self) -> int:
        return len(self.images)




def prepare_dataset_data(cfg):
    # Load train csv
    train_csv = pd.read_csv(
        osp.join(cfg.Dataset.dataset_dir, cfg.Dataset.anno_path))
    # Drop bad instance
    bad_instance_list = [
        '1.2.826.0.1.3680043.20574', '1.2.826.0.1.3680043.8362',
        '1.2.826.0.1.3680043.20756', '1.2.826.0.1.3680043.29952'
    ]
    bad_instance_index = train_csv[train_csv['StudyInstanceUID'].isin(
        bad_instance_list)].index

    train_csv.drop(bad_instance_index, inplace=True)
    train_csv.reset_index(drop=True, inplace=True)

    # Prepare X and y
    uid_list = train_csv['StudyInstanceUID'].to_list()
    images = [
        osp.join(cfg.Dataset.dataset_dir, cfg.Dataset.image_dir, f"{uid}.pt")
        for uid in uid_list
    ]
    labels = train_csv[[
        'patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'
    ]].astype('float32').values

    return images, labels




class RSNADataset2(Dataset):

    def __init__(self, cfg, transform=None):
        super().__init__()
        self.cfg = cfg
        self.transform = transform
        self.images, self.labels = prepare_dataset_data(cfg)
        assert len(self.images) == len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        image = self.images[idx]
        label = self.labels[idx]

        try:
            image = torch.load(image).to(torch.float32)
        except (EOFError, RuntimeError):
            print(f"failed loading: {image}")
        if self.transform is not None:
            image = self.transform(image)
        # in case of predictions, return image name as label
        label = label if label is not None else image
        return (image, label)

    def __len__(self) -> int:
        return len(self.labels)


################################Ignite###########################################
# def build_dataloader(train_transforms, val_transforms, cfg):

#     images, labels = prepare_dataset_data(dataset_dir=cfg.Dataset.dataset_dir)

#     train_images, val_images, train_labels, val_labels = train_test_split(
#         images, labels, test_size=cfg.Dataset.test_size, random_state=42)

#     train_ds = ImageDataset(image_files=train_images,
#                             labels=train_labels,
#                             transform=train_transforms)
#     train_loader = DataLoader(train_ds,
#                               batch_size=cfg.TrainReader.batch_size,
#                               shuffle=cfg.TrainReader.shuffle,)
#                             #   num_workers=cfg.worker_num,)
#                             #   pin_memory=torch.cuda.is_available())
#     val_ds = ImageDataset(image_files=val_images,
#                           labels=val_labels,
#                           transform=val_transforms)
#     val_loader = DataLoader(val_ds,
#                             batch_size=cfg.ValReader.batch_size,)
#                             # num_workers=cfg.worker_num,)
#                             # pin_memory=torch.cuda.is_available())
#     return train_loader, val_loader
