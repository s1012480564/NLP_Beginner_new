# -*- coding: utf-8 -*-
# file: self_explain_robert.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel
# from task3.config_utils import Args
from config_utils import Args
# from task3.layers import SICModel, InterpretationModel
from layers import SICModel, InterpretationModel
from typing import Dict


class ExplainableModel(nn.Module):  # https://arxiv.org/pdf/2012.01786.pdf
    def __init__(self, args: Args):
        super(ExplainableModel, self).__init__()
        self.args = args
        self.roberta_config = RobertaConfig.from_pretrained(args.pretrained_path)
        self.intermediate = RobertaModel.from_pretrained(args.pretrained_path)
        self.span_info_collect = SICModel(self.roberta_config.hidden_size)
        self.interpretation = InterpretationModel(self.roberta_config.hidden_size)
        self.fc = nn.Linear(self.roberta_config.hidden_size, args.num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, start_indexes: torch.Tensor,
                end_indexes: torch.Tensor, span_masks: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[
        str, torch.Tensor]:
        H = self.intermediate(input_ids, attention_mask=attention_mask).last_hidden_state  # B,L,E
        H_ij = self.span_info_collect(H, start_indexes, end_indexes)  # B,S,E  S:span_num
        H, alpha_ij = self.interpretation(H_ij, span_masks)  # B,E
        outputs = self.fc(H)
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss
    from task3.data_utils import preprocess
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from functools import partial
    from task3.data_utils import DataCollator

    args = Args(device=0, num_classes=3, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../../pretrained/roberta-base")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    dataset = load_dataset("json", data_files="../dataset/snli_1.0_test.jsonl")
    dataset = dataset.filter(lambda example: example["gold_label"] in ["contradiction", "neutral", "entailment"])
    dataset = dataset.map(partial(preprocess, tokenizer=tokenizer),
                          remove_columns=["annotator_labels", "captionID", "gold_label", "pairID", "sentence1",
                                          "sentence1_binary_parse", "sentence1_parse", "sentence2",
                                          "sentence2_binary_parse", "sentence2_parse"])

    data_collator = DataCollator(args)
    inputs = data_collator(dataset["train"].to_list()[:4])
    for key in inputs:
        inputs[key] = inputs[key].to(args.device)

    model = ExplainableModel(args).to(args.device)
    print(model(**inputs))
