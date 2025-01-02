# -*- coding: utf-8 -*-
# file: lstm.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from task5.config_utils import Args
from config_utils import Args
from typing import Dict
import pickle


class LSTM(nn.Module):
    def __init__(self, args: Args):
        super(LSTM, self).__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_size, 300)
        self.lstm = nn.LSTM(300, 300, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(300 * 2, args.vocab_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs) -> Dict[
        str, torch.Tensor]:
        lens = torch.sum(input_ids != 0, dim=-1).to("cpu")
        X = self.embed(input_ids)
        X_pack = pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
        X_pack, (_, _) = self.lstm(X_pack)
        X, _ = pad_packed_sequence(X_pack, batch_first=True, total_length=self.args.max_seq_len - 1)
        outputs = self.fc(self.dropout(X))
        loss = None
        if labels is not None:
            loss = self.args.criterion(outputs.view([-1, outputs.shape[-1]]), labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss
    from task5.data_utils import preprocess4glove
    from datasets import load_dataset
    from functools import partial

    args = Args(device=3, dropout=0.1, criterion=CrossEntropyLoss(ignore_index=0), max_seq_len=128)

    tokenizer = pickle.load(open("../glove_pre/poetryTang_tokenizer.dat", 'rb'))
    args.vocab_size = tokenizer.vocab_size

    dataset = load_dataset("text", data_files="../dataset/poetryFromTang.txt", sample_by="paragraph")

    dataset = dataset.map(partial(preprocess4glove, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=["text"])

    features = dataset["train"].features
    inputs = dataset["train"][:4]
    for feature in features:
        inputs[feature] = torch.tensor(inputs[feature]).to(args.device)

    model = LSTM(args).to(args.device)
    print(model(**inputs))
