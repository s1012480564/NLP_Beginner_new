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
from transformers import AutoTokenizer, PreTrainedModel, TrainingArguments, Trainer, EvalPrediction, \
    EarlyStoppingCallback
from functools import partial
from typing import Dict
from config_utils import Args
from data_utils import preprocess, preprocess4glove
from models import RoBERTa, LSTM, CNN
from evaluate import evaluate
import pickle

model_classes = {
    "roberta": RoBERTa,
    "cnn": CNN,
    "lstm": LSTM,
}

dataset_files = {
    "movie_reviews": {
        "train": "dataset/train_split.csv",
        "test": "dataset/val_split.csv"
    },
    "movie_reviews4submission": {
        "train": "dataset/train.csv",
        "test": "dataset/train.csv"
    }
}

pretrained_paths = {
    "roberta": "../../../pretrained/roberta-base",
    "lstm": None,
    "cnn": None,
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


def _compute_metrics(predictions: EvalPrediction, args: Args = None) -> Dict[str, float]:
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = logits.argmax(axis=-1)
    acc = metrics.accuracy_score(labels, preds)
    macro_f1 = metrics.f1_score(labels, preds, labels=range(args.num_classes), average="macro")
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
        np.save(f"outputs/{args.dataset_name}-test_labels.npy", np.array(dataset["test"]["labels"]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="roberta", type=str)
    parser.add_argument("--dataset_name", default="movie_reviews", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--batch_size", default=16, type=int, help="try 16, 32, 64 for BERT models")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="multiple of accumulation steps")
    parser.add_argument("--lr", default=5e-5, type=float, help="try 5e-5, 2e-5 for BERT, 1e-3 for others")
    parser.add_argument("--l2reg", default=1e-5, type=float, help="try 1e-5 for BERT, 1e-2 for others")
    parser.add_argument("--num_epochs", default=3, type=int, help="try larger number for non-BERT models")
    parser.add_argument("--optimizer_name", default="adamw", type=str)
    parser.add_argument("--scheduler_type", default="linear", type=str)
    parser.add_argument("--initializer_name", default="kaiming_uniform", type=str)
    parser.add_argument("--warmup_ratio", default=0.05, type=float, help="try 0.05 for bert-small, 0.1 for large")
    parser.add_argument("--logging_dir", default="logs", type=str)
    parser.add_argument("--logging_steps", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int, help="set seed for reproducibility")
    parser.add_argument("--early_stopping_patience", default=5, type=int)
    parser.add_argument("--num_classes", default=5, type=int)
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
    if args.model_name == "cnn" or args.model_name == "lstm":
        tokenizer = pickle.load(open("glove_pre/movie_reviews_tokenizer.dat", 'rb'))
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_paths[args.model_name])
    model = args.model_class(args)

    dataset_files = dataset_files[args.dataset_name]
    dataset = load_dataset("csv", data_files=dataset_files)
    if args.model_name == "cnn" or args.model_name == "lstm":
        dataset = dataset.map(partial(preprocess4glove, tokenizer=tokenizer), batched=True, batch_size=None,
                              remove_columns=["PhraseId", "SentenceId", "Phrase", "Sentiment"])
    else:
        dataset = dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None,
                              remove_columns=["PhraseId", "SentenceId", "Phrase", "Sentiment"])
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    # 多分类标签高度不平衡会导致深度学习方法准确率明显低下，使用 class_weight 调整损失权重
    # 但是实验发现标准的加权无法解决
    # 数据质量本身可能并不是很好，分析发现最易分类的中性情感准确率也没有特别高，类别加权的意义也不是很大。不是特别适合深度学习方法
    # labels = torch.tensor(dataset["train"]["labels"]).to(args.device)
    # class_weight = torch.bincount(labels, minlength=args.num_classes)
    # class_weight = class_weight.sum() / (class_weight * args.num_classes)
    # args.criterion = nn.CrossEntropyLoss(weight=class_weight)

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
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=partial(_compute_metrics, args=args),
        callbacks=[EarlyStoppingCallback(args.early_stopping_patience)],
    )

    _init_params(model, args)
    trainer.train()

    _save_log(model, dataset, trainer, args, parser_args, save_test_labels=True)