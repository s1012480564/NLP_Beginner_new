# -*- coding: utf-8 -*-
# file: glove_tokenizer.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import numpy as np
import pickle
from typing import List, Dict
import pandas as pd


def pad_and_truncate(tokens, max_len, dtype='int64', value=0):
    x = (np.ones(max_len) * value).astype(dtype)
    trunc = tokens[:max_len]
    trunc = np.asarray(trunc, dtype=dtype)
    x[:len(trunc)] = trunc
    return x


class GloveTokenizer(object):
    def __init__(self):
        self.max_seq_len = 56  # 数据分析结果，训练/测试集最长 56，严格设置节约一点
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 1  # 隐含第0个[PAD]。最后一个[UNK]不算在size中

    def fit(self, sentence):
        words = sentence.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def _transform(self, sentence):
        words = sentence.split()
        unk_idx = self.vocab_size
        tokens = [self.word2idx[w] if w in self.word2idx else unk_idx for w in words]
        if len(tokens) == 0:  # 空白句放一个[UNK]
            tokens = [unk_idx]
        return pad_and_truncate(tokens, self.max_seq_len)

    def batch_encode_plus(self, batch_text: List[str], padding: bool = True, truncation: bool = True) -> List:
        input_ids = []
        for text in batch_text:
            input_ids.append(self._transform(text))
        return input_ids


def build_tokenizer(path: str, save_path: str) -> GloveTokenizer:
    df = pd.read_csv(path)
    document = ''
    for i in range(len(df)):
        phrase = df['Phrase'][i].strip()
        if phrase:
            document += phrase + ' '
    tokenizer = GloveTokenizer()
    tokenizer.fit(document)
    pickle.dump(tokenizer, open(save_path, 'wb'))
    return tokenizer


def build_embedding_matrix(word2idx: Dict[str, int], save_path: str, embed_dim: int = 300) -> np.ndarray:
    embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # [PAD] and [UNK] are all-zeros
    word2vec = {}
    glove_path = "../../../pretrained/glove.840B.300d/glove.840B.300d.txt"

    with open(glove_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        lines = f.readlines()
    for line in lines:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word2vec[word] = np.asarray(vec, dtype='float32')

    for word, i in word2idx.items():
        vec = word2vec.get(word)
        if vec is not None:  # words not found ([UNK]) in embedding index will be all-zeros.
            embedding_matrix[i] = vec

    pickle.dump(embedding_matrix, open(save_path, 'wb'))
    return embedding_matrix


def test_unit():
    # tokenizer = build_tokenizer("dataset/train.csv", save_path="glove_pre/movie_reviews_tokenizer.dat")
    # embedding_matrix = build_embedding_matrix(word2idx=tokenizer.word2idx,
    #                                           save_path="glove_pre/840B_300d_movie_reviews_embedding_matrix.dat")

    tokenizer = pickle.load(open("glove_pre/movie_reviews_tokenizer.dat", 'rb'))
    embedding_matrix = pickle.load(open("glove_pre/840B_300d_movie_reviews_embedding_matrix.dat", 'rb'))

    from datasets import load_dataset
    from functools import partial
    from data_utils import preprocess4glove

    dataset = load_dataset("csv", data_files="dataset/train.csv")
    dataset = dataset.map(partial(preprocess4glove, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'])

    print(dataset["train"])
    print(dataset["train"].features)
    data = dataset["train"][0]
    print(len(data["input_ids"]))
    print(data["labels"])
