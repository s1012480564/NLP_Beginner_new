# -*- coding: utf-8 -*-
# file: inference_composition.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.


from torch import nn
import torch.nn.functional as F
from .bilstm import BiLSTM
# from task3.config_utils import Args
from config_utils import Args
import torch
from typing import List


class InferenceComposition(nn.Module):
    def __init__(self, args: Args):
        super(InferenceComposition, self).__init__()
        self.fc = nn.Linear(4 * 300, 300)
        self.dropout = nn.Dropout(args.dropout)
        self.bilstm = BiLSTM()

    def forward(self, X: torch.Tensor, lens: torch.Tensor | List[int]) -> torch.Tensor:
        X = F.relu(self.fc(X))
        X = self.dropout(X)
        out = self.bilstm(X, lens)
        return out


def test_unit():
    args = Args(dropout=0.5)
    model = InferenceComposition(args)
    X = torch.randn(4, 10, 4 * 300)
    lens = torch.tensor([10, 8, 6, 4])
    out = model(X, lens)
    print(out.shape)
