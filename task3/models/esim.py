# -*- coding: utf-8 -*-
# file: esim.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
import torch.nn as nn
# from task3.config_utils import Args
from config_utils import Args
# from task3.layers import InputEncoding, LocalInferenceModeling, InferenceComposition, Prediction
from layers import InputEncoding, LocalInferenceModeling, InferenceComposition, Prediction
from typing import Dict
import pickle


class ESIM(nn.Module):  # https://arxiv.org/pdf/1609.06038v3.pdf
    def __init__(self, args: Args):
        super(ESIM, self).__init__()
        self.args = args
        # embedding_matrix = pickle.load(open("../glove_pre/840B_300d_snli_embedding_matrix.dat", 'rb'))
        embedding_matrix = pickle.load(open("glove_pre/840B_300d_snli_embedding_matrix.dat", 'rb'))
        self.input_encoding = InputEncoding(embedding_matrix, args)
        self.local_inference = LocalInferenceModeling(embedding_matrix, args)
        self.inference_composition = InferenceComposition(args)
        self.prediction = Prediction(args)

    def forward(self, sentence1_input_ids: torch.Tensor, sentence2_input_ids: torch.Tensor, labels: torch.Tensor) -> \
            Dict[str, torch.Tensor]:
        A, B = sentence1_input_ids, sentence2_input_ids
        lens_A = torch.sum(A != 0, dim=-1).to('cpu')
        lens_B = torch.sum(B != 0, dim=-1).to('cpu')
        A = self.input_encoding(A, lens_A)
        B = self.input_encoding(B, lens_B)
        m_A, m_B = self.local_inference(A, B)
        v_A = self.inference_composition(m_A, lens_A)
        v_B = self.inference_composition(m_B, lens_B)
        outputs = self.prediction(v_A, v_B)
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss
    from task3.data_utils import preprocess4glove
    from datasets import load_dataset
    from functools import partial
    from task3.data_utils import GloveTokenizer

    args = Args(device=0, num_classes=3, dropout=0.1, criterion=CrossEntropyLoss())

    tokenizer = GloveTokenizer()
    dataset = load_dataset("json", data_files="../dataset/snli_1.0_test.jsonl")
    dataset = dataset.filter(lambda example: example["gold_label"] in ["contradiction", "neutral", "entailment"])
    dataset = dataset.map(partial(preprocess4glove, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=["annotator_labels", "captionID", "gold_label", "pairID", "sentence1",
                                          "sentence1_binary_parse", "sentence1_parse", "sentence2",
                                          "sentence2_binary_parse", "sentence2_parse"])

    features = dataset["train"].features
    inputs = dataset["train"][:4]
    for feature in features:
        inputs[feature] = torch.tensor(inputs[feature]).to(args.device)

    model = ESIM(args).to(args.device)
    print(model(**inputs))
