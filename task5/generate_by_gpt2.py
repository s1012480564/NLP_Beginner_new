# -*- coding: utf-8 -*-
# file: generate_by_gpt2.py
# author: songanyang <aysong24@m.fudan.edu.cn>
# Copyright (C) 2024. All Rights Reserved.

import torch
import argparse
from transformers import BertTokenizer, GPT2LMHeadModel
from transformers.generation import GenerationConfig
from random import randint
import warnings
from config_utils import Args
import os


def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'


# 生成随机诗词
def generate(model, args: Args, max_len=64):
    s = ''
    generation_config = GenerationConfig(max_length=max_len, do_sample=True,
                                         eos_token_id=args.tokenizer.sep_token_id,
                                         pad_token_id=args.tokenizer.pad_token_id)
    input_ids = args.tokenizer.encode(s, return_tensors='pt').to(args.device)
    s = args.tokenizer.decode(model.generate(input_ids, generation_config)[0],
                              skip_special_tokens=True)
    s = s.replace(' ', '')
    return s


# beamsearch随机的句子下一个词始终无法满足需求时，手动随机下一个topk的词
def generate_next_random_topk_word(s, model, args: Args, top_k=10):
    tokens_dict = args.tokenizer.encode_plus(s, return_tensors='pt')
    input_ids = tokens_dict['input_ids'].to(args.device)
    attn_mask = tokens_dict['attention_mask'].to(args.device)
    output = model(input_ids, attention_mask=attn_mask).logits
    output = output[0, -1, :]
    sort_idx = torch.sort(output).indices
    word = ''
    while not is_chinese(word):
        word = args.tokenizer.decode(sort_idx[randint(0, top_k - 1)])
    return word


# 生成特定格式的随机古诗，比如五言绝句，k=5个字，n=4句(逗号一句)
def generate_format(model, args: Args, k=5, n=4):
    s = ''
    l = (k + 1) * n
    end, end_pre, repeat = 0, 0, 0
    while end != l:
        generation_config = GenerationConfig(max_length=end + 20, do_sample=True,
                                             eos_token_id=args.tokenizer.sep_token_id,
                                             pad_token_id=args.tokenizer.pad_token_id)
        input_ids = args.tokenizer.encode(s, return_tensors='pt').to(args.device)
        s = args.tokenizer.decode(model.generate(input_ids, generation_config)[0],
                                  skip_special_tokens=True)
        s = s.replace(' ', '')
        for i in range(end, min(l, len(s))):
            sentence_idx = i // (k + 1)
            sentence_word_idx = i % (k + 1)
            if sentence_word_idx == k:
                if sentence_idx % 2 == 0:
                    if s[i] != '，':
                        s = s[:i] + '，'
                        break
                else:
                    if s[i] != '。':
                        s = s[:i] + '。'
                        break
            else:
                if not is_chinese(s[i]):
                    s = s[:i]
                    break
        end = min(l, len(s))
        if end == end_pre:
            repeat += 1
        else:
            end_pre = end
            repeat = 0
        if repeat == 10:
            s += generate_next_random_topk_word(s, model, args)
            end = min(l, len(s))
        s = s[:end]
    s = s.replace('。', '。\n')
    return s


# 生成特定格式的随机藏头诗
def generate_acrostic_format(model, args: Args, head, k=5, n=4):
    assert len(head) == n
    s = ''
    l = (k + 1) * n
    end, end_pre, repeat = 0, 0, 0
    while end != l:
        generation_config = GenerationConfig(max_length=end + 20, do_sample=True,
                                             eos_token_id=args.tokenizer.sep_token_id,
                                             pad_token_id=args.tokenizer.pad_token_id)
        input_ids = args.tokenizer.encode(s, return_tensors='pt').to(args.device)
        s = args.tokenizer.decode(model.generate(input_ids, generation_config)[0],
                                  skip_special_tokens=True)
        s = s.replace(' ', '')
        for i in range(end, min(l, len(s))):
            sentence_idx = i // (k + 1)
            sentence_word_idx = i % (k + 1)
            if sentence_word_idx == 0:
                s = s[:i] + head[sentence_idx]
                break
            elif sentence_word_idx == k:
                if sentence_idx % 2 == 0:
                    if s[i] != '，':
                        s = s[:i] + '，'
                        break
                else:
                    if s[i] != '。':
                        s = s[:i] + '。'
                        break
            else:
                if not is_chinese(s[i]):
                    s = s[:i]
                    break
        end = min(l, len(s))
        if end == end_pre:
            repeat += 1
        else:
            end_pre = end
            repeat = 0
        if repeat == 10:
            s += generate_next_random_topk_word(s, model, args)
            end = min(l, len(s))
        s = s[:end]
    s = s.replace('。', '。\n')
    return s


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)

    args = Args(device=0, pretrained_path="../../../pretrained/gpt2-chinese-poem")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    args.device = 0

    args.tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)

    model = GPT2LMHeadModel.from_pretrained(args.pretrained_path)

    model = model.to(args.device)
    model.eval()

    with torch.no_grad():
        print(generate(model, args))
        print()
        print(generate(model, args))
        print()
        print(generate_format(model, args))
        print(generate_format(model, args, k=7))
        print(generate_format(model, args, n=8))
        print(generate_format(model, args, k=7, n=8))
        print(generate_acrostic_format(model, args, '春夏秋冬'))
        print(generate_acrostic_format(model, args, '春夏秋冬', k=7))
        print(generate_acrostic_format(model, args, '上下左右东南西北', n=8))
        print(generate_acrostic_format(model, args, '上下左右东南西北', k=7, n=8))
