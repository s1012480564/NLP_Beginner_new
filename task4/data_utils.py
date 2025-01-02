from transformers import PreTrainedTokenizer
from typing import Dict, List
from datasets import ClassLabel
from glove_tokenizer import GloveTokenizer

# 对 padding，CEloss 可以 ignore_index=0。CRF 算 loss 是传 mask，所以 padding_idx=0 和 0类别 混在一起了也没关系，所以我这里这么做了
# 但是大概还是设置一个单独的 padding_idx=0 比较好，其实写起来可能更简单，可阅读性也更好
class_label = ClassLabel(names=["B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER", "O"])


def _pad4labels(labels: List[List[int]], max_seq_length: int, pad4glove: bool) -> List[List[int]]:
    for i in range(len(labels)):
        if pad4glove:
            labels[i] = labels[i] + (max_seq_length - len(labels[i])) * [0]
        else:
            labels[i] = [0] + labels[i] + (max_seq_length - len(labels[i]) - 1) * [0]
    return labels


def preprocess(examples: Dict[str, List], tokenizer: PreTrainedTokenizer) -> Dict[str, List]:
    batch_input_ids = []
    batch_labels_ids = []
    for paragraph in examples["text"]:
        input_words, label_words = [], []
        lines = paragraph.strip().split("\n")
        for line in lines:
            input_word, _, _, label_word = line.split()
            input_words.append(input_word)
            label_words.append(label_word)
        batch_input_ids.append(tokenizer.encode(input_words, add_special_tokens=False))
        batch_labels_ids.append(class_label.str2int(label_words))
    examples["input_ids"], _, examples["attention_mask"] = tokenizer.batch_encode_plus(batch_input_ids, padding=True,
                                                                                       truncation=True,
                                                                                       is_split_into_words=True).values()
    examples["labels"] = _pad4labels(batch_labels_ids, max_seq_length=len(examples["input_ids"][0]), pad4glove=False)
    return examples


def preprocess4glove(examples: Dict[str, List], tokenizer: GloveTokenizer) -> Dict[str, List]:
    batch_input_words = []
    batch_labels_ids = []
    for paragraph in examples["text"]:
        input_words, label_words = [], []
        lines = paragraph.strip().split("\n")
        for line in lines:
            input_word, _, _, label_word = line.split()
            input_words.append(input_word)
            label_words.append(label_word)
        batch_input_words.append(input_words)
        batch_labels_ids.append(class_label.str2int(label_words))
    examples["input_ids"] = tokenizer.batch_encode_plus(batch_input_words)
    examples["labels"] = _pad4labels(batch_labels_ids, max_seq_length=124, pad4glove=True)
    return examples


def test_unit():
    from transformers import BertTokenizer
    from datasets import load_dataset
    from functools import partial

    tokenizer = BertTokenizer.from_pretrained("../../../pretrained/bert-base-cased")
    dataset = load_dataset("text", data_files="dataset/train.txt", sample_by="paragraph")
    dataset = dataset.filter(lambda example: example["text"] != "-DOCSTART- -X- -X- O")

    dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=["text"])

    print(dataset["train"])
    print(dataset["train"].features)
    data = dataset["train"][0]
    print(len(data["input_ids"]))
    print(data["labels"])
    print(len(data["labels"]))
