# -*- coding: utf-8 -*-
# file: roberta.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
# from task2.config_utils import Args
from config_utils import Args
from typing import Dict


class RoBERTa(nn.Module):
    def __init__(self, args: Args):
        super(RoBERTa, self).__init__()
        self.args = args
        self.roberta_config = RobertaConfig.from_pretrained(args.pretrained_path)
        self.roberta = RobertaModel.from_pretrained(args.pretrained_path)
        self.fc = nn.Linear(self.roberta_config.hidden_size, args.num_classes)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[
        str, torch.Tensor]:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        outputs = self.fc(self.dropout(out))
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss
    from task2.data_utils import preprocess
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from functools import partial

    args = Args(device=0, num_classes=5, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../../pretrained/roberta-base")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    dataset = load_dataset("csv", data_files="../dataset/train.csv")
    dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'])

    features = dataset["train"].features
    inputs = dataset["train"][:4]
    for feature in features:
        inputs[feature] = torch.tensor(inputs[feature]).to(args.device)

    model = RoBERTa(args).to(args.device)
    print(model(**inputs))
