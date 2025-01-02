# -*- coding: utf-8 -*-
# file: sic_model.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn


class SICModel(nn.Module):
    def __init__(self, hidden_size: int):
        super(SICModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, start_indexs: torch.Tensor,
                end_indexs: torch.Tensor) -> torch.Tensor:
        W1_h = self.fc1(hidden_states)  # B,L,E
        W2_h = self.fc2(hidden_states)
        W3_h = self.fc3(hidden_states)
        W4_h = self.fc4(hidden_states)

        W1_hi = torch.index_select(W1_h, dim=1, index=start_indexs)  # B,S,E  S:span_num
        W2_hj = torch.index_select(W2_h, dim=1, index=end_indexs)
        W3_hi = torch.index_select(W3_h, dim=1, index=start_indexs)
        W3_hj = torch.index_select(W3_h, dim=1, index=end_indexs)
        W4_hi = torch.index_select(W4_h, dim=1, index=start_indexs)
        W4_hj = torch.index_select(W4_h, dim=1, index=end_indexs)

        H_ij = torch.tanh(W1_hi + W2_hj + (W3_hi - W3_hj) + W4_hi * W4_hj)  # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        return H_ij


def test_unit():
    model = SICModel(hidden_size=300)
    hidden_states = torch.randn(2, 7, 300)
    start_indexs = torch.tensor([0, 1, 2, 3, 4, 5])
    end_indexs = torch.tensor([1, 2, 3, 4, 5, 6])
    H_ij = model(hidden_states, start_indexs, end_indexs)
    print(H_ij.shape)  # [2, 3, 300]
