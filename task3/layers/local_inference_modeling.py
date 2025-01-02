# -*- coding: utf-8 -*-
# file: local_inference_modeling.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch import nn
import torch.nn.functional as F
# from task3.config_utils import Args
from config_utils import Args
import numpy as np


class LocalInferenceModeling(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, args: Args):
        super(LocalInferenceModeling, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        score = A @ (B.transpose(1, 2))  # [B,LA,LB]
        scoreA = F.softmax(score, dim=2)
        scoreB = F.softmax(score, dim=1)
        A_t = scoreA @ B  # [B,LA,E]
        B_t = scoreB.transpose(1, 2) @ A  # [B,LB,E]
        out_A = torch.cat([A, A_t, A - A_t, A * A_t], dim=-1)  # [B,LA,4E]
        out_B = torch.cat([B, B_t, B - B_t, B * B_t], dim=-1)  # [B,LB,4E]
        return out_A, out_B

def test_unit():
    A = torch.rand(2, 3, 4)
    B = torch.rand(2, 5, 4)
    model = LocalInferenceModeling(None, Args())
    out_A, out_B = model(A, B)
    print(out_A.shape, out_B.shape)