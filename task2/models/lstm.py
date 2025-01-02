# -*- coding: utf-8 -*-
# file: lstm.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
# from task2.config_utils import Args
from config_utils import Args
from typing import Dict
import pickle


class LSTM(nn.Module):
    def __init__(self, args: Args):
        super(LSTM, self).__init__()
        self.args = args
        # embedding_matrix = pickle.load(open("../glove_pre/840B_300d_movie_reviews_embedding_matrix.dat", 'rb'))
        embedding_matrix = pickle.load(open("glove_pre/840B_300d_movie_reviews_embedding_matrix.dat", 'rb'))
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = nn.LSTM(300, 300, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(300 * 2, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        lens = torch.sum(input_ids != 0, dim=-1).to("cpu")
        X = self.embed(input_ids)  # [B,L,E]
        X_pack = pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(X_pack)  # [2,B,E]
        h_n = torch.cat((h_n[0], h_n[1]), dim=-1)  # [B,2E]
        outputs = self.fc(self.dropout(h_n))
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss
    from task2.data_utils import preprocess4glove
    from datasets import load_dataset
    from functools import partial

    args = Args(device=0, num_classes=5, dropout=0.1, criterion=CrossEntropyLoss())

    tokenizer = pickle.load(open("../glove_pre/movie_reviews_tokenizer.dat", 'rb'))
    dataset = load_dataset("csv", data_files="../dataset/train.csv")
    dataset = dataset.map(partial(preprocess4glove, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'])

    features = dataset["train"].features
    inputs = dataset["train"][:4]
    for feature in features:
        inputs[feature] = torch.tensor(inputs[feature]).to(args.device)

    model = LSTM(args).to(args.device)
    print(model(**inputs))
