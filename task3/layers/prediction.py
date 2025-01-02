# -*- coding: utf-8 -*-
# file: prediction.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch import nn
import torch.nn.functional as F
# from task3.config_utils import Args
from config_utils import Args


class Prediction(nn.Module):
    def __init__(self, args: Args):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * 78, 78),
            nn.Tanh(),
            nn.Linear(78, args.num_classes)
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_avg = F.avg_pool1d(A, A.shape[2]).squeeze(-1)  # [B,LA]
        A_max = F.max_pool1d(A, A.shape[2]).squeeze(-1)  # [B,LA]
        B_avg = F.avg_pool1d(B, B.shape[2]).squeeze(-1)  # [B,LB]
        B_max = F.max_pool1d(B, B.shape[2]).squeeze(-1)  # [B,LB]
        V = torch.cat((A_avg, A_max, B_avg, B_max), dim=-1)
        out = self.mlp(V)
        return out


def test_unit():
    args = Args(num_classes=3)
    model = Prediction(args)
    A = torch.randn(2, 78, 300)
    B = torch.randn(2, 78, 300)
    out = model(A, B)
    print(out.shape)
