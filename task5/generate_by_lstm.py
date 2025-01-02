# -*- coding: utf-8 -*-
# file: generate_by_lstm.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
from random import randint
from models import LSTM
from config_utils import Args
import os
import pickle
from safetensors.torch import load_file


def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'


# 生成随机诗词
def generate(model, tokenizer, k=100, max_len=128):
    # 第一个词对topk采样
    out = model(torch.tensor(tokenizer._transform('')).reshape(1, -1))["logits"][0]
    out = out[0]
    sort_idx = torch.sort(out, descending=True).indices
    s = ''
    while not is_chinese(s):
        s = tokenizer.idx2word[sort_idx[randint(0, k - 1)].item()]
    for i in range(max_len - 1):
        out = model(torch.tensor(tokenizer._transform(s)).reshape(1, -1))["logits"][0]
        out = out[i + 1]
        out = torch.argmax(out, dim=-1)
        word = tokenizer.idx2word[out.item()]
        if word == '[EOS]':
            break
        s += word
    return s


# 生成特定格式的随机古诗，比如五言绝句，k=5个字，n=4句(逗号一句)
def generate_format(model, tokenizer, k=5, n=4, top_k=100):
    s = ''
    for i in range(n):
        for j in range(k):
            out = model(torch.tensor(tokenizer._transform(s)).reshape(1, -1))["logits"][0]
            out = out[len(s)]
            sort_idx = torch.sort(out, descending=True).indices
            word = ''
            if i == 0 and j == 0:
                while not is_chinese(word):
                    word = tokenizer.idx2word[sort_idx[randint(0, top_k - 1)].item()]
            else:
                for idx in sort_idx:
                    word = tokenizer.idx2word[idx.item()]
                    if is_chinese(word):
                        break
            s += word
        if i % 2 == 0:
            s += '，'
        else:
            s += '。\n'
    return s


# 生成特定格式的随机藏头诗，topk=0生成固定藏头诗
def generate_acrostic_format(model, tokenizer, head, k=5, n=4, top_k=10):
    assert len(head) == n
    s = ''
    for i in range(n):
        for j in range(k):
            if j == 0:
                s += head[i]
                continue
            out = model(torch.tensor(tokenizer._transform(s)).reshape(1, -1))["logits"][0]
            out = out[len(s)]
            sort_idx = torch.sort(out, descending=True).indices
            word = ''
            if j == 1:
                while not is_chinese(word):
                    word = tokenizer.idx2word[sort_idx[randint(0, top_k - 1)].item()]
            else:
                for idx in sort_idx:
                    word = tokenizer.idx2word[idx.item()]
                    if is_chinese(word):
                        break
            s += word
        if i % 2 == 0:
            s += '，'
        else:
            s += '。\n'
    return s


if __name__ == '__main__':
    args = Args(dropout=0.1, max_seq_len=128)

    tokenizer = pickle.load(open("glove_pre/poetryTang_tokenizer.dat", 'rb'))
    args.vocab_size = tokenizer.vocab_size

    model = LSTM(args)

    state_dict = load_file("outputs/lstm/checkpoint-500/model.safetensors")
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        print(generate(model, tokenizer))
        print()
        print(generate(model, tokenizer))
        print()
        print(generate_format(model, tokenizer))
        print(generate_format(model, tokenizer, k=7))
        print(generate_format(model, tokenizer, n=8))
        print(generate_format(model, tokenizer, k=7, n=8))
        print(generate_acrostic_format(model, tokenizer, '春夏秋冬'))
        print(generate_acrostic_format(model, tokenizer, '春夏秋冬', k=7))
        print(generate_acrostic_format(model, tokenizer, '上下左右东南西北', n=8))
        print(generate_acrostic_format(model, tokenizer, '上下左右东南西北', k=7, n=8))
