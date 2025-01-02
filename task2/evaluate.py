import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import Dataset
from sklearn import metrics
from typing import Dict
from config_utils import Args
from time import strftime, localtime


def evaluate(model: nn.Module, test_dataset: Dataset, args: Args, save_test_preds: bool = True) -> Dict[str, float]:
    test_dataset = test_dataset.with_format("torch", device=args.device)
    data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    labels_all, preds_all = None, None
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader):
            logits = model(**batch)["logits"]
            labels = batch["labels"]
            preds = logits.argmax(dim=-1)
            labels_all = torch.cat([labels_all, labels]) if labels_all is not None else labels
            preds_all = torch.cat([preds_all, preds]) if preds_all is not None else preds
    labels_all = labels_all.cpu().numpy()
    preds_all = preds_all.cpu().numpy()
    if save_test_preds:
        np.save(
            f"{args.output_dir}/{args.model_name}-{args.dataset_name}-{strftime("%y%m%d-%H%M", localtime())}-test_preds.npy",
            preds_all)
    acc = round(metrics.accuracy_score(labels_all, preds_all), 4)
    macro_f1 = round(metrics.f1_score(labels_all, preds_all, labels=range(args.num_classes), average="macro"), 4)
    return {"test_accuracy": acc, "test_macro_f1": macro_f1}


def test_unit():
    from models import RoBERTa
    from torch.nn import CrossEntropyLoss
    from data_utils import preprocess
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from functools import partial

    args = Args(model_name="roberta", dataset_name="movie_reviews", device=0, output_dir="outputs/roberta",
                batch_size=16, num_classes=5, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/roberta-base")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)
    test_dataset = load_dataset("csv", data_files="dataset/val_split.csv")
    test_dataset = test_dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None,
                                    remove_columns=['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'])
    test_dataset = test_dataset["train"]

    model = RoBERTa(args).to(args.device)

    print(evaluate(model, test_dataset, args))
