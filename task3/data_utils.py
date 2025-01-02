from transformers import PreTrainedTokenizerFast, RobertaTokenizer
from typing import Dict, List, Any
from datasets import ClassLabel
from glove_tokenizer import GloveTokenizer
import torch
from config_utils import Args
from prompt_constructor import PromptConstructor

class_label = ClassLabel(names=["contradiction", "neutral", "entailment"])


class DataCollator:
    def __init__(self, args: Args):
        self.args = args
        self.tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_path)

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        batch = {}
        input_ids = [f["input_ids"] for f in features]
        input_ids, attention_mask = self.tokenizer.batch_encode_plus(input_ids, padding=True,
                                                                     is_split_into_words=True, truncation=True,
                                                                     return_tensors='pt').values()
        input_ids = input_ids.to(self.args.device)
        attention_mask = attention_mask.to(self.args.device)

        max_len = input_ids.shape[1]
        start_indexes = []
        end_indexes = []
        for i in range(1, max_len - 1):
            for j in range(i, max_len - 1):
                start_indexes.append(i)
                end_indexes.append(j)
        start_indexes = torch.tensor(start_indexes, device=self.args.device)
        end_indexes = torch.tensor(end_indexes, device=self.args.device)
        lens = torch.sum(attention_mask != 0, dim=-1)

        lens_mat = lens.reshape(-1, 1)
        middle_indexes_mat = torch.where(input_ids == 2)[1][::2].reshape(-1, 1)
        start_indexes_ex = start_indexes.expand((input_ids.shape[0], -1))
        end_indexes_ex = end_indexes.expand((input_ids.shape[0], -1))

        span_masks_bool = (start_indexes_ex <= lens_mat - 2) & (end_indexes_ex <= lens_mat - 2) & (
                (start_indexes_ex > middle_indexes_mat) | (end_indexes_ex < middle_indexes_mat))
        span_masks = torch.ones(input_ids.shape[0], start_indexes.shape[0], device=self.args.device) * 1e6 * (
            ~span_masks_bool)

        batch["labels"] = torch.tensor([f["labels"] for f in features])
        batch["input_ids"] = input_ids.cpu()
        batch["attention_mask"] = attention_mask.cpu()
        batch["start_indexes"] = start_indexes.cpu()
        batch["end_indexes"] = end_indexes.cpu()
        batch["span_masks"] = span_masks.cpu()

        return batch


def preprocess(example: Dict[str, Any], tokenizer: PreTrainedTokenizerFast) -> Dict[str, Any]:
    sentence1, sentence2 = example["sentence1"], example["sentence2"]
    if sentence1.endswith("."):
        sentence1 = sentence1[:-1]
    if sentence2.endswith("."):
        sentence2 = sentence2[:-1]
    sentence1_input_ids = tokenizer.encode(sentence1, add_special_tokens=False)
    sentence2_input_ids = tokenizer.encode(sentence2, add_special_tokens=False)
    example["input_ids"] = sentence1_input_ids + [2] + sentence2_input_ids
    example["labels"] = class_label.str2int(example["gold_label"])
    return example


def preprocess4glove(examples: Dict[str, List], tokenizer: GloveTokenizer) -> Dict[str, List]:
    examples["sentence1_input_ids"] = tokenizer.batch_encode_plus(examples["sentence1"], padding=True, truncation=True)
    examples["sentence2_input_ids"] = tokenizer.batch_encode_plus(examples["sentence2"], padding=True, truncation=True)
    examples["labels"] = list(class_label.str2int(examples["gold_label"]))
    return examples


def preprocess4inference(examples: Dict[str, List], class_label: ClassLabel, pc: PromptConstructor,
                         demonstration_examples: List[List[Dict[str, str]]] = None) -> Dict[str, List]:
    examples["prompts"] = []
    for i in range(len(examples["sentence1"])):
        if demonstration_examples is not None:
            pc.set_examples(demonstration_examples[i])
        for j in range(class_label.num_classes):
            examples["prompts"].append(pc.get_prompt(
                {"premise": examples["sentence1"][i], "hypothesis": examples["sentence2"][i],
                 "relationship": class_label.int2str(j)},
                same_format_as_example=True))
    return examples


def test_unit():
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from functools import partial
    import example_generator

    # tokenizer = AutoTokenizer.from_pretrained("../../../pretrained/roberta-base")
    dataset = load_dataset("json", data_files="dataset/snli_1.0_test.jsonl")
    dataset = dataset.filter(lambda example: example["gold_label"] in ["contradiction", "neutral", "entailment"])
    # dataset = dataset.map(partial(preprocess, tokenizer=tokenizer),
    #                       remove_columns=["annotator_labels", "captionID", "gold_label", "pairID", "sentence1",
    #                                       "sentence1_binary_parse", "sentence1_parse", "sentence2",
    #                                       "sentence2_binary_parse", "sentence2_parse"])
    #
    # print(dataset["train"])
    # print(dataset["train"].features)
    # data = dataset["train"][0]
    # print(len(data["input_ids"]))
    # print(data["labels"])
    #
    # args = Args(device=0, pretrained_path="../../../pretrained/roberta-base")
    # data_collator = DataCollator(args)
    # print(data_collator(dataset["train"].to_list()[:2]))

    labels = dataset["train"]["gold_label"]
    class_label = ClassLabel(names=["Contradiction", "Neutral", "Entailment"])

    pc = PromptConstructor()
    pc.set_template("NLI")
    pc.set_examples(example_generator.get_constant_artificial_examples())

    dataset = dataset.map(partial(preprocess4inference, class_label=class_label, pc=pc), batched=True, batch_size=None,
                          remove_columns=["annotator_labels", "captionID", "gold_label", "pairID", "sentence1",
                                          "sentence1_binary_parse", "sentence1_parse", "sentence2",
                                          "sentence2_binary_parse", "sentence2_parse"])

    print(dataset["train"])
    print(dataset["train"].features)
    data = dataset["train"][0]
    print(data["prompts"])
    print()
    print(f"{data["prompts"]!r}")
    print(len(labels))
    print(labels[0])
