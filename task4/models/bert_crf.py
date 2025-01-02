# -*- coding: utf-8 -*-
# file: bert_crf.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
from torch import nn
from transformers import BertConfig, BertModel
from TorchCRF import CRF
# from task4.config_utils import Args
from config_utils import Args
from typing import Dict, List


class BERT_CRF(nn.Module):
    def __init__(self, args: Args):
        super(BERT_CRF, self).__init__()
        self.args = args
        self.bert_config = BertConfig.from_pretrained(args.pretrained_path)
        self.bert = BertModel.from_pretrained(args.pretrained_path)
        self.fc = nn.Linear(self.bert_config.hidden_size, args.num_classes)
        self.crf = CRF(args.num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[
        str, torch.Tensor]:
        out = self.bert(input_ids, attention_mask).last_hidden_state
        logits = self.fc(out)
        logits = logits[:, 1:, :]  # remove [CLS] token
        labels = labels[:, 1:]  # remove [CLS] token
        mask = attention_mask.bool()[:, 1:]  # remove [CLS] token
        lens = mask.sum(dim=-1)
        mask[range(mask.shape[0]), lens - 1] = False
        loss = -self.crf(logits, labels, mask).mean()
        return {"loss": loss, "logits": logits}

    def predict(self, logits: torch.FloatTensor, input_ids: torch.Tensor, **kwargs) -> List[List[int]]:
        mask = input_ids != 0
        lens = torch.sum(input_ids != 0, dim=-1)
        mask = mask[:, 1:]  # remove [CLS] token
        mask[range(mask.shape[0]), lens - 2] = False  # remove [SEP] token
        predictions = self.crf.viterbi_decode(logits, mask)
        return predictions


def test_unit():
    from task4.data_utils import preprocess
    from transformers import BertTokenizer
    from datasets import load_dataset
    from functools import partial

    args = Args(device=0, num_classes=9, dropout=0.1, pretrained_path="../../../../pretrained/bert-base-cased")

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)
    dataset = load_dataset("text", data_files="../dataset/train.txt", sample_by="paragraph")
    dataset = dataset.filter(lambda example: example["text"] != "-DOCSTART- -X- -X- O")

    dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=["text"])

    features = dataset["train"].features
    inputs = dataset["train"][:4]
    for feature in features:
        inputs[feature] = torch.tensor(inputs[feature]).to(args.device)

    model = BERT_CRF(args).to(args.device)
    outputs = model(**inputs)
    print(outputs)

    print(model.predict(outputs["logits"], **inputs))
