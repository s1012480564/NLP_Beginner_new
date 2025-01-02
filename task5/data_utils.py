from typing import Dict, List
from glove_tokenizer import GloveTokenizer
import numpy as np


def preprocess4glove(examples: Dict[str, List], tokenizer: GloveTokenizer) -> Dict[str, List]:
    text_ids = np.array(tokenizer.batch_encode_plus(examples["text"]))
    examples["input_ids"] = text_ids[:, :-1]
    examples["labels"] = text_ids[:, 1:]
    return examples


def test_unit():
    pass
    # from transformers import BertTokenizer
    # from datasets import load_dataset
    # from functools import partial
    #
    # tokenizer = BertTokenizer.from_pretrained("../../../pretrained/bert-base-cased")
    # dataset = load_dataset("text", data_files="dataset/train.txt", sample_by="paragraph")
    # dataset = dataset.filter(lambda example: example["text"] != "-DOCSTART- -X- -X- O")
    #
    # dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None,
    #                       remove_columns=["text"])
    #
    # print(dataset["train"])
    # print(dataset["train"].features)
    # data = dataset["train"][0]
    # print(len(data["input_ids"]))
    # print(data["labels"])
    # print(len(data["labels"]))
