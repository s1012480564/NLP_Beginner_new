import numpy as np
import argparse
import os
from datasets import load_dataset, ClassLabel
import vllm
from vllm import LLM, SamplingParams
from typing import List
from config_utils import Args
from prompt_constructor import PromptConstructor
from data_utils import preprocess4inference
import example_generator
from example_generator import get_knn_example_idx
from functools import partial
from transformers import BartTokenizerFast, BartModel

dataset_files = {
    "snli": {
        "train": "dataset/snli_1.0_train.jsonl",
        "val": "dataset/snli_1.0_dev.jsonl",
        "test": "dataset/snli_1.0_test.jsonl"
    }
}

model_paths = {
    "llama-3.2-1b-instruct": "../../../pretrained/llama-3.2-1b-instruct",
}


def calculate_logppls(outputs: List[vllm.RequestOutput], num_examples: int, num_classes: int,
                      output_ids_lens: List[int], args: Args) -> np.ndarray:
    logppls = np.zeros((num_examples, num_classes))
    for i in range(num_examples):
        for j in range(num_classes):
            prompt_logprobs = outputs[i * num_classes + j].prompt_logprobs
            output_ids_len = output_ids_lens[i] if "channel" in args.method_name else output_ids_lens[j]
            for k in range(1, output_ids_len + 1):
                logppls[i][j] += -list(prompt_logprobs[-k].values())[0].logprob
            logppls[i][j] /= output_ids_len
    return logppls


def inference(llm: LLM, prompts: List[str], sampling_params: SamplingParams, num_examples: int,
              output_ids_lens: List[int], logppls_bias: np.ndarray, args: Args) -> np.ndarray:
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    logppls = calculate_logppls(outputs, num_examples, args.num_classes, output_ids_lens, args=args)
    if "calibrate" in args.method_name:
        logppls -= logppls_bias
    preds = np.argmin(logppls, axis=-1)
    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name", default="KATE", type=str)
    parser.add_argument("--model_name", default="llama-3.2-1b-instruct", type=str)
    parser.add_argument("--dataset_name", default="snli", type=str)
    parser.add_argument("--template_name", default="NLI", type=str)
    parser.add_argument("--instruction", default="", type=str, help="e.g. \"instruction\" or empty str \"\" ")
    parser.add_argument("--device", default="0,1,2,3,4,5,6,7", type=str, help="e.g. \"0,1\" ")
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--seed", default=42, type=int, help="set seed for reproducibility")
    parser.add_argument("--gpu_memory_utilization", default=0.6, type=float)

    parser_args = parser.parse_args()
    args = Args(**vars(parser_args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_path = model_paths[args.model_name]
    class_label = ClassLabel(names=["Contradiction", "Neutral", "Entailment"])
    args.num_classes = len(class_label.names)

    dataset_files = dataset_files[args.dataset_name]
    dataset = load_dataset("json", data_files=dataset_files)
    dataset = dataset.filter(lambda example: example["gold_label"] in ["contradiction", "neutral", "entailment"])
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    num_examples = len(test_dataset)
    labels = test_dataset["gold_label"]
    class_label_uncased = ClassLabel(names=["contradiction", "neutral", "entailment"])
    label_ids = class_label_uncased.str2int(labels)
    np.save(f"outputs/{args.dataset_name}-test_labels.npy", label_ids)

    pc = PromptConstructor()
    pc.set_template(args.template_name)
    pc.set_examples(example_generator.get_constant_artificial_examples())
    # pc.set_examples(example_generator.get_constant_artificial_examples_6shot())

    # if args.instruction:
    # pc.set_prefix(
    #     "Now I will ask you to answer the relationship between the input premise and hypothesis. Please answer my " + \
    #     "question based on the following samples of premise, hypothesis and relationship.\nIn this question, " + \
    #     "the relationship between the premise and hypothesis includes three categories: Contradiction, Neutral, " + \
    #     "Entailment. Contradiction means that the premise and hypothesis are contradictory, that is, the premise " + \
    #     "can clearly infer that the hypothesis is not true. Neutral means that there is no clear direct " + \
    #     "connection between the premise and the hypothesis, that is, the premise can neither clearly infer that " + \
    #     "the hypothesis is true nor clearly infer that the hypothesis is not true. Entailment means that the " + \
    #     "hypothesis is included in the premise, that is, the premise can clearly infer that the hypothesis is " + \
    #     "true. What you need to do is to answer the relationship between the input premise and hypothesis.\n")
    # pc.set_infix(
    #     "The following is the formal input. Please imitate the above samples and answer my question as required:\n")

    # pc.set_prefix(
    #     "Now I will ask you to answer the relationship between the input premise and hypothesis. Please answer my " + \
    #     "question based on the following samples of premise, hypothesis and relationship.\nIn this question, " + \
    #     "the relationship between the premise and hypothesis includes three categories: Contradiction, Neutral, " + \
    #     "Entailment, namely, whether the premise can infer the hypothesis. " + \
    #     "What you need to do is to answer the relationship between the input premise and hypothesis.\n")
    # pc.set_infix(
    #     "The following is the formal input. Please imitate the above samples and answer my question as required:\n")

    # pc.set_prefix(
    #     "Now I will ask you to answer the relationship between the input premise and hypothesis. Please answer my " + \
    #     "question based on the following samples of premise, hypothesis and relationship.\nIn this question, " + \
    #     "the relationship between the premise and hypothesis includes three categories: Contradiction, Neutral, " + \
    #     "Entailment, namely, whether the premise can infer the hypothesis.\n")
    # pc.set_infix(
    #     "The following is the formal input. Please imitate the above samples and answer my question as required:\n")

    # pc.set_prefix(
    #     "Now I will ask you to answer the relationship between the input premise and hypothesis. Please answer my " + \
    #     "question based on the following samples of premise, hypothesis and relationship.\nIn this question, " + \
    #     "the relationship between the premise and hypothesis includes three categories: Contradiction, Neutral, " + \
    #     "Entailment, namely, whether the premise can infer the hypothesis.\n")

    # pc.set_prefix(
    #     "Now I will ask you to answer the relationship between the input premise and hypothesis. Please answer my " + \
    #     "question based on the following samples of premise, hypothesis and relationship.\n")

    # 这个。。意外发现只加这一句在这个任务效果挺好，但是我 require 都去掉了啊。误打误撞啊这，在 task2 试了下并不好
    # 太玄学了，理解不能。不敢加了，还是不加了。。。
    # pc.set_infix(
    #     "The following is the formal input. Please imitate the above samples and answer my question as required:\n")

    # pc.set_infix(
    #     "The following is the formal input. Please imitate the above samples and answer my question:\n")

    test_dataset_temp = test_dataset.map(lambda examples: examples, batched=True, batch_size=None,
                                         remove_columns=["annotator_labels", "captionID", "gold_label", "pairID",
                                                         "sentence1_binary_parse", "sentence1_parse",
                                                         "sentence2_binary_parse", "sentence2_parse"])

    demonstration_examples = None
    if args.method_name == "KATE":
        pretrained_path = "../../../pretrained/bart-large"
        tokenizer = BartTokenizerFast.from_pretrained(pretrained_path)
        model = BartModel.from_pretrained(pretrained_path).to(0)
        neigh_idx = get_knn_example_idx(test_dataset, tokenizer, model, "retriever/knn_fit.dat")
        premise = np.array(train_dataset["sentence1"])
        hypothesis = np.array(train_dataset["sentence2"])
        relationship = np.array(train_dataset["gold_label"])
        demonstration_examples = example_generator.get_knn_examples(neigh_idx, premise, hypothesis, relationship)

    test_dataset = test_dataset.map(
        partial(preprocess4inference, class_label=class_label, pc=pc, demonstration_examples=demonstration_examples),
        batched=True, batch_size=None, remove_columns=["annotator_labels", "captionID", "gold_label", "pairID",
                                                       "sentence1",
                                                       "sentence1_binary_parse", "sentence1_parse", "sentence2",
                                                       "sentence2_binary_parse", "sentence2_parse"])

    prompts = test_dataset["prompts"]

    sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1)
    llm = LLM(model=model_path, seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization)

    tokenizer = llm.get_tokenizer()
    output_ids_lens = []
    logppls_bias = None
    if "channel" in args.method_name:
        output_ids_lens = test_dataset_temp.map(lambda example: {"length": len(
            tokenizer.encode("Premise: " + example["sentence1"] + "\nHypothesis: " + example["sentence2"],
                             add_special_tokens=False))})["length"]

        if "calibrate" in args.method_name:
            null_input_prompts = test_dataset_temp.map(lambda examples: {"null_input_prompts": [
                pc.get_null_input_prompt({"premise": sentence1, "hypothesis": sentence2}, same_format_as_example=True)
                for sentence1, sentence2 in zip(examples["sentence1"], examples["sentence2"])]}, batched=True,
                                                       batch_size=None)["null_input_prompts"]
            outputs = llm.generate(null_input_prompts, sampling_params=sampling_params)
            logppls_bias = calculate_logppls(outputs, num_examples, 1, output_ids_lens, args)
            logppls_bias = logppls_bias.repeat(args.num_classes, axis=-1)

    else:
        for i in range(args.num_classes):
            output_ids_lens.append(len(tokenizer.encode(class_label.int2str(i), add_special_tokens=False)))

        if "calibrate" in args.method_name:
            null_input_prompts = [pc.get_null_input_prompt({"relationship": name}, same_format_as_example=True) for name
                                  in class_label.names]
            outputs = llm.generate(null_input_prompts, sampling_params=sampling_params)
            logppls_bias = calculate_logppls(outputs, 1, args.num_classes, output_ids_lens, args)
            logppls_bias = logppls_bias.repeat(num_examples, axis=0)

    preds = inference(llm, prompts=prompts, sampling_params=sampling_params, num_examples=num_examples,
                      output_ids_lens=output_ids_lens, logppls_bias=logppls_bias, args=args)

    np.save(
        f"{args.output_dir}/{args.method_name}-{args.model_name}-{args.dataset_name}-{args.template_name}{"-" + args.instruction if args.instruction else ""}-test_preds.npy",
        preds)
