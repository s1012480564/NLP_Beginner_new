# -*- coding: utf-8 -*-
# file: lstm_crf.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch import nn
from TorchCRF import CRF
# from task4.layers import BiLSTM
from layers import BiLSTM
# from task4.config_utils import Args
from config_utils import Args
import pickle
from typing import Dict, List


class LSTM_CRF(nn.Module):  # https://arxiv.org/pdf/1603.01360.pdf
    def __init__(self, args: Args):
        super(LSTM_CRF, self).__init__()
        self.args = args
        # embedding_matrix = pickle.load(open("../glove_pre/840B_300d_ner_embedding_matrix.dat", 'rb'))
        embedding_matrix = pickle.load(open("glove_pre/840B_300d_ner_embedding_matrix.dat", 'rb'))
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bilstm = BiLSTM()
        self.fc = nn.Linear(100, args.num_classes)
        self.crf = CRF(args.num_classes)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        mask = input_ids != 0
        lens = torch.sum(input_ids != 0, dim=-1).to("cpu")
        X = self.embed(input_ids)
        out = self.bilstm(X, lens)
        logits = self.fc(out)
        loss = -self.crf(logits, labels, mask).mean()
        return {"loss": loss, "logits": logits}

    def predict(self, logits: torch.FloatTensor, input_ids: torch.Tensor, **kwargs) -> List[List[int]]:
        mask = input_ids != 0
        out = self.crf.viterbi_decode(logits, mask)
        return out


def test_unit():
    from task4.data_utils import preprocess4glove
    from datasets import load_dataset
    from functools import partial

    args = Args(device=0, num_classes=9, dropout=0.1, pretrained_path="../../../../pretrained/bert-base-cased")

    tokenizer = pickle.load(open("../glove_pre/ner_tokenizer.dat", 'rb'))

    dataset = load_dataset("text", data_files="../dataset/train.txt", sample_by="paragraph")
    dataset = dataset.filter(lambda example: example["text"] != "-DOCSTART- -X- -X- O")
    dataset = dataset.map(partial(preprocess4glove, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=["text"])

    features = dataset["train"].features
    inputs = dataset["train"][:4]
    for feature in features:
        inputs[feature] = torch.tensor(inputs[feature]).to(args.device)

    model = LSTM_CRF(args).to(args.device)
    outputs = model(**inputs)
    print(outputs)

    print(model.predict(outputs["logits"], **inputs))
