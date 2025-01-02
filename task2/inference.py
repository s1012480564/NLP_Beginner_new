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
from functools import partial
from transformers import BartTokenizerFast, BartModel
from example_generator import get_knn_example_idx

dataset_files = {
    "movie_reviews": {
        "train": "dataset/train.csv",
        "test": "dataset/test.csv"
    },
    "movie_reviews_val": {
        "train": "dataset/train_split.csv",
        "test": "dataset/val_split.csv"
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


def inference(llm: LLM, prompts: List[str], sampling_params: SamplingParams, class_label: ClassLabel,
              num_examples: int, output_ids_lens: List[int], logppls_bias: np.ndarray, args: Args) -> np.ndarray:
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
    parser.add_argument("--dataset_name", default="movie_reviews", type=str)
    parser.add_argument("--template_name", default="sentiment", type=str)
    parser.add_argument("--instruction", default="", type=str, help="e.g. \"instruction\" or empty str \"\" ")
    parser.add_argument("--device", default="0,1,2,3,4,5,6,7", type=str, help="e.g. \"0,1\" ")
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--seed", default=42, type=int, help="set seed for reproducibility")
    parser.add_argument("--gpu_memory_utilization", default=0.6, type=float)

    parser_args = parser.parse_args()
    args = Args(**vars(parser_args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_path = model_paths[args.model_name]
    class_label = ClassLabel(names=["Negative", "Somewhat Negative", "Neutral", "Somewhat Positive", "Positive"])
    args.num_classes = len(class_label.names)

    dataset_files = dataset_files[args.dataset_name]
    dataset = load_dataset("csv", data_files=dataset_files)
    train_dataset = dataset["train"]
    # test_dataset = dataset["test"] if args.method_name == "KATE" else dataset["train"]
    #
    # num_examples = len(test_dataset)
    # labels = np.array(test_dataset["Sentiment"])
    # np.save(f"outputs/{args.dataset_name}-" + ("val" if args.method_name == "KATE" else "train") + "_labels.npy",
    #         labels)

    pc = PromptConstructor()
    pc.set_template(args.template_name)
    pc.set_examples(example_generator.get_constant_artificial_examples())
    # pc.set_examples(example_generator.get_variable_artificial_examples_10shot())

    # if args.instruction:
    #     pc.set_infix(
    #         "The following is the formal input. Please imitate the above samples and answer my question as required:\n")

    # test_dataset_temp = test_dataset.map(lambda examples: examples, batched=True, batch_size=None,
    #                                      remove_columns=["PhraseId", "SentenceId", "Sentiment"])
    #
    # demonstration_examples = None
    # if args.method_name == "KATE":
    #     pretrained_path = "../../../pretrained/bart-large"
    #     tokenizer = BartTokenizerFast.from_pretrained(pretrained_path)
    #     model = BartModel.from_pretrained(pretrained_path).to(0)
    #     neigh_idx = get_knn_example_idx(test_dataset, tokenizer, model, "retriever/knn_fit.dat")
    #     input = np.array(train_dataset["Phrase"])
    #     output = np.array(train_dataset["Sentiment"])
    #     demonstration_examples = example_generator.get_knn_examples(neigh_idx, input, output)
    #
    # test_dataset = test_dataset.map(
    #     partial(preprocess4inference, class_label=class_label, pc=pc, demonstration_examples=demonstration_examples),
    #     batched=True, batch_size=None, remove_columns=["PhraseId", "SentenceId", "Phrase", "Sentiment"])
    #
    # prompts = test_dataset["prompts"]

    sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1)
    llm = LLM(model=model_path, seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization)

    # tokenizer = llm.get_tokenizer()
    # output_ids_lens = []
    # logppls_bias = None
    # if "channel" in args.method_name:
    #     output_ids_lens = test_dataset_temp.map(lambda example: {"length": len(
    #         tokenizer.encode("Sentence: " + example["Phrase"], add_special_tokens=False))})["length"]
    #
    #     if "calibrate" in args.method_name:
    #         null_input_prompts = test_dataset_temp.map(lambda examples: {
    #             "null_input_prompts": [pc.get_null_input_prompt({"input": phrase}, same_format_as_example=True) for
    #                                    phrase in examples["Phrase"]]}, batched=True, batch_size=None)[
    #             "null_input_prompts"]
    #         outputs = llm.generate(null_input_prompts, sampling_params=sampling_params)
    #         logppls_bias = calculate_logppls(outputs, num_examples, 1, output_ids_lens, args)
    #         logppls_bias = logppls_bias.repeat(args.num_classes, axis=-1)
    #
    # else:
    #     for i in range(args.num_classes):
    #         output_ids_lens.append(len(tokenizer.encode(class_label.int2str(i), add_special_tokens=False)))
    #
    #     if "calibrate" in args.method_name:
    #         null_input_prompts = [pc.get_null_input_prompt({"output": name}, same_format_as_example=True) for name
    #                               in class_label.names]
    #         outputs = llm.generate(null_input_prompts, sampling_params=sampling_params)
    #         logppls_bias = calculate_logppls(outputs, 1, args.num_classes, output_ids_lens, args)
    #         logppls_bias = logppls_bias.repeat(num_examples, axis=0)
    #
    # preds = inference(llm, prompts=prompts, sampling_params=sampling_params, class_label=class_label,
    #                   num_examples=num_examples, output_ids_lens=output_ids_lens, logppls_bias=logppls_bias, args=args)
    #
    # np.save(
    #     f"{args.output_dir}/{args.method_name}-{args.model_name}-{args.dataset_name}-{args.template_name}{"-" + args.instruction if args.instruction else ""}-" + (
    #         "val" if args.method_name == "KATE" else "train") + "_preds.npy",
    #     preds)

    test_dataset = load_dataset("csv", data_files=dataset_files["test"])
    test_dataset = test_dataset["train"]
    num_examples = len(test_dataset)

    test_dataset_temp = test_dataset.map(lambda examples: examples, batched=True, batch_size=None,
                               remove_columns=["PhraseId", "SentenceId"])

    demonstration_examples = None
    if args.method_name == "KATE":
        pretrained_path = "../../../pretrained/bart-large"
        tokenizer = BartTokenizerFast.from_pretrained(pretrained_path)
        model = BartModel.from_pretrained(pretrained_path).to(0)
        neigh_idx = get_knn_example_idx(test_dataset, tokenizer, model, "retriever/knn_fit.dat")
        input = np.array(train_dataset["Phrase"])
        output = np.array(train_dataset["Sentiment"])
        demonstration_examples = example_generator.get_knn_examples(neigh_idx, input, output)

    test_dataset = test_dataset.map(
        partial(preprocess4inference, class_label=class_label, pc=pc, demonstration_examples=demonstration_examples),
        batched=True, batch_size=None,remove_columns=["PhraseId", "SentenceId", "Phrase"])

    prompts = test_dataset["prompts"]

    output_ids_lens = []
    logppls_bias = None
    if "channel" in args.method_name:
        output_ids_lens = test_dataset_temp.map(lambda example: {"length": len(
            tokenizer.encode("Sentence: " + example["Phrase"], add_special_tokens=False))})["length"]

        if "calibrate" in args.method_name:
            null_input_prompts = test_dataset_temp.map(lambda examples: {
                "null_input_prompts": [pc.get_null_input_prompt({"input": phrase}, same_format_as_example=True) for
                                       phrase in examples["Phrase"]]}, batched=True, batch_size=None)[
                "null_input_prompts"]
            outputs = llm.generate(null_input_prompts, sampling_params=sampling_params)
            logppls_bias = calculate_logppls(outputs, num_examples, 1, output_ids_lens, args)
            logppls_bias = logppls_bias.repeat(class_label.num_classes, axis=-1)
    else:
        for i in range(class_label.num_classes):
            output_ids_lens.append(len(tokenizer.encode(class_label.int2str(i), add_special_tokens=False)))

        if "calibrate" in args.method_name:
            null_input_prompts = [pc.get_null_input_prompt({"output": name}, same_format_as_example=True) for name
                                  in class_label.names]
            outputs = llm.generate(null_input_prompts, sampling_params=sampling_params)
            logppls_bias = calculate_logppls(outputs, 1, class_label.num_classes, output_ids_lens, args)
            logppls_bias = logppls_bias.repeat(num_examples, axis=0)

    preds = inference(llm, prompts=prompts, sampling_params=sampling_params, class_label=class_label,
                      num_examples=num_examples, output_ids_lens=output_ids_lens, logppls_bias=logppls_bias, args=args)

    np.save(
        f"{args.output_dir}/{args.method_name}-{args.model_name}-{args.dataset_name}-{args.template_name}{"-" + args.instruction if args.instruction else ""}-test_preds.npy",
        preds)
