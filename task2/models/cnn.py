# -*- coding: utf-8 -*-
# file: cnn.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
# from task2.config_utils import Args
from config_utils import Args
from typing import Dict
import pickle


class CNN(nn.Module):  # https://arxiv.org/pdf/1408.5882.pdf
    def __init__(self, args: Args):
        super(CNN, self).__init__()
        self.args = args
        # embedding_matrix = pickle.load(open("../glove_pre/840B_300d_movie_reviews_embedding_matrix.dat", 'rb'))
        embedding_matrix = pickle.load(open("glove_pre/840B_300d_movie_reviews_embedding_matrix.dat", 'rb'))
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(k, 300)) for k in [3, 4, 5]
        ])
        self.fc = nn.Linear(300, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        X = self.embed(input_ids)  # [B, L, E]
        X = X.unsqueeze(1)  # [B, C=1, L, E]
        Xs = [F.relu(conv(X)).squeeze(3) for conv in self.convs]  # [B, C, L - k + 1] for each
        Xs = [F.max_pool1d(X, X.shape[2]).squeeze(2) for X in Xs]  # [B, C] for each
        X = self.dropout(torch.cat(Xs, dim=1))  # [B,C*len(ks)]
        outputs = self.fc(X)
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

    model = CNN(args).to(args.device)
    print(model(**inputs))
