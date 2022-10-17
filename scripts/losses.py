#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-11 20:10:16
LastEditors  : ChenjieXu
LastEditTime : 2022-10-12 19:45:59
FilePath     : /cervical/scripts/losses.py
Description  : 
'''
import torch

from torch.nn.modules.loss import _Loss


class CustomWithLogitsLoss(_Loss):

    def __init__(self) -> None:

        super().__init__()
        self.competition_weights = {
            'negative': torch.tensor([7, 1, 1, 1, 1, 1, 1, 1],
                                     dtype=torch.float),
            'positive': torch.tensor([14, 2, 2, 2, 2, 2, 2, 2],
                                     dtype=torch.float),
        }

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        weights =  self.competition_weights['positive'].type_as(input) * target + \
                   self.competition_weights['negative'].type_as(input) * ( 1 - target)
        loss = torch.nn.BCEWithLogitsLoss(reduction='none')(input, target.type_as(input))
        loss = torch.mul(loss, weights)
        loss = torch.div(torch.sum(loss, dim=1), torch.sum(weights, dim=1))
        loss = torch.mean(loss)

        return loss

class CustomLoss(_Loss):

    def __init__(self) -> None:

        super().__init__()
        self.competition_weights = {
            'negative': torch.tensor([7, 1, 1, 1, 1, 1, 1, 1],
                                     dtype=torch.float),
            'positive': torch.tensor([14, 2, 2, 2, 2, 2, 2, 2],
                                     dtype=torch.float),
        }

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        
        weights =  self.competition_weights['positive'].type_as(input) * target + \
                   self.competition_weights['negative'].type_as(input) * ( 1 - target)
        with torch.cuda.amp.autocast(enabled=False):
            loss = torch.nn.BCELoss(reduction='none')(input, target.type_as(input))
        loss = torch.mul(loss, weights)
        loss = torch.div(torch.sum(loss, dim=1), torch.sum(weights, dim=1))
        loss = torch.mean(loss)

        return loss