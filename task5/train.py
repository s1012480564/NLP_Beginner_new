import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, kaiming_normal_, xavier_uniform_, xavier_normal_
import os
import math
import argparse
from time import strftime, localtime
from datasets import load_dataset, Dataset
from transformers import PreTrainedModel, TrainingArguments, Trainer, EarlyStoppingCallback
from functools import partial
from config_utils import Args
from data_utils import preprocess4glove
from models import LSTM
import pickle

model_classes = {
    "lstm": LSTM
}

dataset_files = {
    "poetryTang": "dataset/poetryFromTang.txt"
}

pretrained_paths = {
    "lstm": None,
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


def _save_log(model: nn.Module, dataset: Dataset, trainer: Trainer, args: Args, parser_args) -> None:
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
        file.write(f"> num_examples: {len(dataset["train"])}\n")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="lstm", type=str)
    parser.add_argument("--dataset_name", default="poetryTang", type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--batch_size", default=164, type=int, help="try 16, 32, 64 for BERT models")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="multiple of accumulation steps")
    parser.add_argument("--lr", default=1e-3, type=float, help="try 5e-5, 2e-5 for BERT, 1e-3 for others")
    parser.add_argument("--l2reg", default=1e-2, type=float, help="try 1e-5 for BERT, 1e-2 for others")
    parser.add_argument("--num_epochs", default=500, type=int, help="try larger number for non-BERT models")
    parser.add_argument("--optimizer_name", default="adamw", type=str)
    parser.add_argument("--scheduler_type", default="constant", type=str)
    parser.add_argument("--initializer_name", default="kaiming_uniform", type=str)
    parser.add_argument("--warmup_ratio", default=0, type=float, help="try 0.05 for bert-small, 0.1 for large")
    parser.add_argument("--logging_dir", default="logs", type=str)
    parser.add_argument("--logging_steps", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int, help="set seed for reproducibility")
    parser.add_argument("--dropout", default=0.1, type=float, help="try 0.1 for pretrained models, 0.5 for others")
    parser.add_argument("--max_seq_len", default=128, type=int)

    parser_args = parser.parse_args()
    args = Args(**vars(parser_args))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    args.device = 0

    args.model_class = model_classes[args.model_name]
    args.initializer = initializer_funcs[args.initializer_name]
    args.criterion = nn.CrossEntropyLoss(ignore_index=0)
    args.pretrained_path = pretrained_paths[args.model_name]

    tokenizer = pickle.load(open("glove_pre/poetryTang_tokenizer.dat", 'rb'))
    args.vocab_size = tokenizer.vocab_size

    model = args.model_class(args)

    dataset_files = dataset_files[args.dataset_name]
    dataset = load_dataset("text", data_files=dataset_files, sample_by="paragraph")

    dataset = dataset.map(partial(preprocess4glove, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=["text"])

    train_dataset = dataset["train"]

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
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )

    _init_params(model, args)
    trainer.train()

    _save_log(model, dataset, trainer, args, parser_args)
