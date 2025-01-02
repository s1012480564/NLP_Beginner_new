# -*- coding: utf-8 -*-
# file: input_encoding.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch import nn
from .bilstm import BiLSTM
# from task3.config_utils import Args
from config_utils import Args
import numpy as np
from typing import List


class InputEncoding(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, args: Args):
        super(InputEncoding, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(args.dropout)
        self.bilstm = BiLSTM()

    def forward(self, X: torch.Tensor, lens: torch.Tensor | List[int]) -> torch.Tensor:
        lens = torch.sum(X != 0, dim=-1).to('cpu')
        X = self.embed(X)  # [B, L, E]
        X = self.dropout(X)
        out = self.bilstm(X, lens)  # [B, L, E]
        return out


def test_unit():
    args = Args(dropout=0.5)
    embedding_matrix = np.random.rand(100, 300)
    model = InputEncoding(embedding_matrix, args)
    X = torch.randint(0, 100, (4, 10))
    lens = [5, 7, 3, 9]
    out = model(X, lens)
    print(out.shape)
