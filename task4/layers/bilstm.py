# -*- coding: utf-8 -*-
# file: lstm.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from typing import List


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.max_seq_len = 124
        self.bilstm = nn.LSTM(300, 100 // 2, 2, batch_first=True, bidirectional=True)

    def forward(self, X: torch.Tensor, lens: torch.Tensor | List[int]) -> torch.Tensor:
        X_pack = pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)  # [B, L, E]
        out, (_, _) = self.bilstm(X_pack)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=self.max_seq_len)  # [B, L, E]
        return out


def test_unit():
    import torch
    model = BiLSTM()
    X = torch.rand(3, 124, 300)
    lens = [124, 70, 60]
    out = model(X, lens)
    print(out.shape)  # [3, 124, 300]
