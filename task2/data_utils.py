from transformers import PreTrainedTokenizerFast
from typing import Dict, List
from glove_tokenizer import GloveTokenizer
from prompt_constructor import PromptConstructor
from datasets import ClassLabel
import example_generator


def preprocess(examples: Dict[str, List], tokenizer: PreTrainedTokenizerFast) -> Dict[str, List]:
    examples["input_ids"], examples["attention_mask"] = tokenizer.batch_encode_plus(examples["Phrase"], padding=True,
                                                                                    truncation=True).values()
    examples["labels"] = examples["Sentiment"]
    return examples


def preprocess4glove(examples: Dict[str, List], tokenizer: GloveTokenizer) -> Dict[str, List]:
    examples["input_ids"] = tokenizer.batch_encode_plus(examples["Phrase"], padding=True, truncation=True)
    examples["labels"] = examples["Sentiment"]
    return examples


def preprocess4inference(examples: Dict[str, List], class_label: ClassLabel, pc: PromptConstructor,
                         demonstration_examples: List[List[Dict[str, str]]] = None) -> Dict[str, List]:
    examples["prompts"] = []
    for i in range(len(examples["Phrase"])):
        if demonstration_examples is not None:
            pc.set_examples(demonstration_examples[i])
        for j in range(class_label.num_classes):
            examples["prompts"].append(pc.get_prompt({"input": examples["Phrase"][i], "output": class_label.int2str(j)},
                                                     same_format_as_example=True))
    return examples


def test_unit():
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from functools import partial

    # tokenizer = AutoTokenizer.from_pretrained("../../../pretrained/roberta-base")
    dataset = load_dataset("csv", data_files="dataset/train.csv")
    # dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None,
    #                       remove_columns=['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'])
    #
    # print(dataset["train"])
    # print(dataset["train"].features)
    # data = dataset["train"][0]
    # print(len(data["input_ids"]))
    # print(data["labels"])

    labels = dataset["train"]["Sentiment"]
    class_label = ClassLabel(names=["Negative", "Somewhat Negative", "Neutral", "Somewhat Positive", "Positive"])

    pc = PromptConstructor()
    pc.set_template("origin_with_space")
    pc.set_examples(example_generator.get_constant_artificial_examples())

    dataset = dataset.map(partial(preprocess4inference, class_label=class_label, pc=pc), batched=True, batch_size=None,
                          remove_columns=['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'])

    print(dataset["train"])
    print(dataset["train"].features)
    data = dataset["train"][0]
    print(data["prompts"])
    print()
    print(f"{data["prompts"]!r}")
    print(len(labels))
    print(labels[0])
