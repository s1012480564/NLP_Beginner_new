# -*- coding: utf-8 -*-
# file: glove_tokenizer.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import numpy as np
import pickle
from typing import List
from datasets import load_dataset
from config_utils import Args


def pad_and_truncate(tokens, max_len, dtype='int64', value=0):
    x = (np.ones(max_len) * value).astype(dtype)
    trunc = tokens[:max_len]
    trunc = np.asarray(trunc, dtype=dtype)
    x[:len(trunc)] = trunc
    return x


class GloveTokenizer(object):
    def __init__(self, args: Args):
        self.max_seq_len = args.max_seq_len
        self.word2idx = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2}
        self.idx2word = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]'}
        self.vocab_size = 3  # [PAD]:0，[BOS]:1，[EOS]:2。最后一个[UNK]

    def fit(self, sentence):
        words = []
        for word in sentence:
            words.append(word)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        self.word2idx['[UNK]'] = self.vocab_size
        self.idx2word[self.vocab_size] = '[UNK]'
        self.vocab_size += 1

    def _transform(self, sentence):
        words = []
        for word in sentence:
            words.append(word)
        words = ['[BOS]'] + words + ['[EOS]']
        unk_idx = self.vocab_size - 1
        tokens = [self.word2idx[w] if w in self.word2idx else unk_idx for w in words]
        return pad_and_truncate(tokens, self.max_seq_len)

    def batch_encode_plus(self, batch_text: List[str], padding: bool = True, truncation: bool = True) -> List:
        input_ids = []
        for text in batch_text:
            input_ids.append(self._transform(text))
        return input_ids


def build_tokenizer(path: str, save_path: str, args: Args) -> GloveTokenizer:
    dataset = load_dataset("text", data_files=path, sample_by="document")
    document = dataset["train"]["text"][0]
    tokenizer = GloveTokenizer(args)
    tokenizer.fit(document)
    pickle.dump(tokenizer, open(save_path, 'wb'))
    return tokenizer


def test_unit():
    from functools import partial
    from data_utils import preprocess4glove
    # args = Args(max_seq_len=128)
    # tokenizer = build_tokenizer("dataset/poetryFromTang.txt", save_path="glove_pre/poetryTang_tokenizer.dat",
    #                             args=args)

    tokenizer = pickle.load(open("glove_pre/poetryTang_tokenizer.dat", 'rb'))

    dataset = load_dataset("text", data_files="dataset/poetryFromTang.txt", sample_by="paragraph")

    dataset = dataset.map(partial(preprocess4glove, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=["text"])

    print(dataset["train"])
    print(dataset["train"].features)
    data = dataset["train"][0]
    print(len(data["input_ids"]))
    print(len(data["labels"]))
