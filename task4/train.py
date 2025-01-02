import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, kaiming_normal_, xavier_uniform_, xavier_normal_
from sklearn import metrics
import os
import math
import argparse
from time import strftime, localtime
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, PreTrainedModel, TrainingArguments, Trainer, EvalPrediction, \
    EarlyStoppingCallback
from functools import partial
from typing import Dict
from config_utils import Args
from data_utils import preprocess, preprocess4glove
from models import LSTM_CRF, BERT_CRF
from evaluate import evaluate
import pickle

model_classes = {
    "lstm_crf": LSTM_CRF,
    "bert_crf": BERT_CRF
}

dataset_files = {
    "ner": {
        "train": "dataset/train.txt",
        "val": "dataset/dev.txt",
        "test": "dataset/test.txt"
    }
}

pretrained_paths = {
    "bert_crf": "../../../pretrained/bert-base-cased",
    "lstm_crf": None,
}

initializer_funcs = {
    'kaiming_uniform': kaiming_uniform_,
    'kaiming_normal': kaiming_normal_,
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
}


def _init_params(model: nn.Module, args: Args) -> None:
    for child in model.children():
        if isinstance(child, PreTrainedModel):  # skip PreTrainedModel params
            continue
        for p in child.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    args.initializer(p)
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    nn.init.uniform_(p, a=-stdv, b=stdv)


def _compute_metrics(predictions: EvalPrediction, args: Args = None, model: nn.Module = None) -> Dict[str, float]:
    logits = predictions.predictions
    labels = predictions.label_ids
    input_ids = predictions.inputs
    preds = model.predict(torch.tensor(logits).to(args.device), torch.tensor(input_ids).to(args.device))
    lens = np.sum(input_ids != 0, axis=-1)
    labels_all, preds_all = [], []
    for i in range(len(labels)):
        label_ids = None
        if args.model_name == "lstm_crf":
            label_ids = labels[i][:lens[i]].tolist()
        elif args.model_name == "bert_crf":
            label_ids = labels[i][1:lens[i] - 1].tolist()
        labels_all += label_ids
    for pred in preds:
        preds_all += pred
    labels_all = np.array(labels_all)
    preds_all = np.array(preds_all)
    acc = metrics.accuracy_score(labels_all, preds_all)
    macro_f1 = metrics.f1_score(labels_all, preds_all, labels=range(args.num_classes), average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def _save_log(model: nn.Module, dataset: Dataset, trainer: Trainer, args: Args, parser_args,
              save_test_labels: bool = False, save_test_preds: bool = True) -> None:
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            n_trainable_params += p.numel()
        else:
            n_nontrainable_params += p.numel()

    len_dataloader = math.ceil(len(dataset["train"]) / parser_args.batch_size)
    num_update_steps_per_epoch = len_dataloader // parser_args.gradient_accumulation_steps
    total_optimization_steps = math.ceil(parser_args.num_epochs * num_update_steps_per_epoch)

    log_file = f"logs/{args.model_name}-{args.dataset_name}-{strftime("%y%m%d-%H%M", localtime())}.log"
    with open(log_file, "w") as file:
        file.write(f"cuda_max_memory_allocated: {torch.cuda.max_memory_allocated()}\n")
        file.write(
            f"> num_trainable_parameters: {n_trainable_params}, num_nontrainable_parameters: {n_nontrainable_params}\n")
        file.write(f"> num_train_examples: {len(dataset["train"])}, num_test_examples: {len(dataset["test"])}\n")
        file.write(f"> total_optimization_steps: {total_optimization_steps}\n")
        file.write("> training arguments: \n")
        for arg in vars(parser_args):
            file.write(f">>> {arg}: {getattr(parser_args, arg)}\n")

        epoch = 0
        file.write('>' * 100 + '\n')
        file.write(f"epoch: {epoch}\n")
        for log in trainer.state.log_history:
            for key in log:
                if key != "learning_rate":
                    log[key] = round(log[key], 4)

            if "eval_loss" in log:
                file.write(f"> {log}\n")
                epoch += 1
                if epoch < parser_args.num_epochs:
                    file.write('>' * 100 + '\n')
                    file.write(f"epoch: {epoch}\n")
            else:
                file.write(f"{log}\n")

        test_results = evaluate(model, dataset["test"], args, save_test_preds)
        file.write(f">>> {test_results}\n")

    if save_test_labels:
        dataset_np = dataset["test"].with_format("numpy")
        lens = np.sum(dataset_np["input_ids"] != 0, axis=-1)
        labels_all = np.array([])
        for i, label_ids in enumerate(dataset_np["labels"]):
            if args.model_name == "lstm_crf":
                label_ids = label_ids[:lens[i]]
            elif args.model_name == "bert_crf":
                label_ids = label_ids[1:lens[i] - 1]  # remove [cls] and [sep] token
            labels_all = np.concatenate((labels_all, label_ids))
        np.save(f"outputs/{args.dataset_name}-test_labels.npy", labels_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert_crf", type=str)
    parser.add_argument("--dataset_name", default="ner", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--batch_size", default=16, type=int, help="try 16, 32, 64 for BERT models")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="multiple of accumulation steps")
    parser.add_argument("--lr", default=2e-5, type=float, help="try 5e-5, 2e-5 for BERT, 1e-3 for others")
    parser.add_argument("--l2reg", default=1e-5, type=float, help="try 1e-5 for BERT, 1e-2 for others")
    parser.add_argument("--num_epochs", default=4, type=int, help="try larger number for non-BERT models")
    parser.add_argument("--optimizer_name", default="adamw", type=str)
    parser.add_argument("--scheduler_type", default="linear", type=str)
    parser.add_argument("--initializer_name", default="kaiming_uniform", type=str)
    parser.add_argument("--warmup_ratio", default=0.05, type=float, help="try 0.05 for bert-small, 0.1 for large")
    parser.add_argument("--logging_dir", default="logs", type=str)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int, help="set seed for reproducibility")
    parser.add_argument("--early_stopping_patience", default=10, type=int)
    parser.add_argument("--num_classes", default=9, type=int)
    parser.add_argument("--dropout", default=0.1, type=float, help="try 0.1 for pretrained models, 0.5 for others")

    parser_args = parser.parse_args()
    args = Args(**vars(parser_args))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    args.device = 0

    args.model_class = model_classes[args.model_name]
    args.initializer = initializer_funcs[args.initializer_name]
    args.criterion = nn.CrossEntropyLoss()
    args.pretrained_path = pretrained_paths[args.model_name]

    tokenizer = None
    if args.model_name == "lstm_crf":
        tokenizer = pickle.load(open("glove_pre/ner_tokenizer.dat", 'rb'))
    elif args.model_name == "bert_crf":
        tokenizer = BertTokenizer.from_pretrained(pretrained_paths[args.model_name])
    model = args.model_class(args)

    dataset_files = dataset_files[args.dataset_name]
    dataset = load_dataset("text", data_files=dataset_files, sample_by="paragraph")

    dataset = dataset.filter(lambda example: example["text"] != "-DOCSTART- -X- -X- O")

    if args.model_name == "lstm_crf":
        dataset = dataset.map(partial(preprocess4glove, tokenizer=tokenizer), batched=True, batch_size=None,
                              remove_columns=["text"])
    elif args.model_name == "bert_crf":
        dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None,
                              remove_columns=["text"])
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.l2reg,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type=args.scheduler_type,
        warmup_ratio=args.warmup_ratio,
        log_level="info",
        logging_dir=args.logging_dir,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        include_for_metrics=["inputs"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=partial(_compute_metrics, args=args, model=model),
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience)],
    )

    _init_params(model, args)
    trainer.train()

    _save_log(model, dataset, trainer, args, parser_args, save_test_labels=True)
