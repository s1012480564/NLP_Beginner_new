# -*- coding: utf-8 -*-
# file: interpretation_model.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
from typing import Tuple


class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super(InterpretationModel, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, h_ij: torch.Tensor, span_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        o_ij = self.fc(h_ij).squeeze(-1)  # B,S  S:span_num
        o_ij = o_ij - span_masks  # mask illegal span, e^(-1e6)≈0
        alpha_ij = nn.functional.softmax(o_ij, dim=1)  # softmax归一化得打分向量，然后利用学习到的ij间关系，对全部hij加权为一个向量
        H = (alpha_ij.unsqueeze(1) @ h_ij).squeeze(1)  # (B,1,S) @ (B,S,E) = (B,E)
        return H, alpha_ij

def test_unit():
    model = InterpretationModel(hidden_size=300)
    h_ij = torch.rand(3, 5, 300)
    span_masks = torch.zeros(3, 5)
    span_masks[0, 2] = 1
    span_masks[1, 3] = 1
    span_masks[2, 4] = 1
    H, alpha_ij = model(h_ij, span_masks)
    print(H.shape)
    print(alpha_ij.shape)