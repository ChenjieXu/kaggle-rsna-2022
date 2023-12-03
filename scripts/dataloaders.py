
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
LastEditTime: 2023-12-02 19:11:58
FilePath     : /cervical/scripts/dataloaders.py
Description  : 
'''

import os
import os.path as osp
from typing import Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

from scripts.factory import build_transforms


class RSNADataset(Dataset):
    def __init__(self, images, labels, transform=None):
        super().__init__()
        self.transform = transform
        self.images = images
        self.labels = labels
        assert len(self.images) == len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = torch.tensor(image_path).to(torch.float32)
        except (EOFError, RuntimeError):
            print(f"Failed loading: {image_path}")

        if self.transform is not None:
            image = self.transform(image)

        # 在进行预测时，将图像名称作为标签返回。
        label = label if label is not None else image_path
        return image, label

    def __len__(self) -> int:
        return len(self.labels)


def prepare_dataset_data(cfg):
    train_csv = pd.read_csv(osp.join(cfg.Dataset.dataset_dir, cfg.Dataset.anno_path))

    # 删除有问题的数据
    bad_instance_list = ['1.2.826.0.1.3680043.20574', '1.2.826.0.1.3680043.8362',
                         '1.2.826.0.1.3680043.20756', '1.2.826.0.1.3680043.29952']
    bad_instance_index = train_csv[train_csv['StudyInstanceUID'].isin(bad_instance_list)].index
    train_csv.drop(bad_instance_index, inplace=True)
    train_csv.reset_index(drop=True, inplace=True)

    uid_list = train_csv['StudyInstanceUID'].to_list()
    images = [osp.join(cfg.Dataset.dataset_dir, cfg.Dataset.image_dir, f"{uid}.pt") for uid in uid_list]
    labels = train_csv[['patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].astype('float32').values

    return images, labels


class RSNADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_transform, self.val_transform = build_transforms(self.cfg)

    def setup(self, stage: Optional[str] = None):
        images, labels = prepare_dataset_data(self.cfg)
        # 使用 train_test_split 进行数据集划分
        train_images, val_images, train_labels, val_labels = train_test_split(
            images,
            labels,
            test_size=self.cfg.Dataset.test_size,
            random_state=42,
        )
        # 初始化训练和验证数据集
        self.rsna_train = RSNADataset(train_images, train_labels, self.train_transform)
        self.rsna_val = RSNADataset(val_images, val_labels, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.rsna_train,
            batch_size=self.cfg.TrainReader.batch_size,
            shuffle=self.cfg.TrainReader.shuffle,
            num_workers=self.cfg.TrainReader.num_workers,
            pin_memory=self.cfg.TrainReader.pin_memory,
            drop_last=self.cfg.TrainReader.drop_last
        )

    def val_dataloader(self):
        return DataLoader(
            self.rsna_val,
            batch_size=self.cfg.ValReader.batch_size,
            shuffle=self.cfg.ValReader.shuffle,
            num_workers=self.cfg.ValReader.num_workers,
            pin_memory=self.cfg.ValReader.pin_memory,
            drop_last=self.cfg.ValReader.drop_last
        )