import torch
from tqdm import tqdm
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
    labels_all, preds_all = [], []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(data_loader)):
            logits = model(**batch)["logits"]
            labels = batch["labels"]
            lens = torch.sum(batch["input_ids"] != 0, dim=-1)
            preds = model.predict(logits, **batch)
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
    if save_test_preds:
        np.save(
            f"{args.output_dir}/{args.model_name}-{args.dataset_name}-{strftime("%y%m%d-%H%M", localtime())}-test_preds.npy",
            preds_all)
    acc = round(metrics.accuracy_score(labels_all, preds_all), 4)
    macro_f1 = round(metrics.f1_score(labels_all, preds_all, labels=range(args.num_classes), average="macro"), 4)
    return {"test_accuracy": acc, "test_macro_f1": macro_f1}


def test_unit():
    from models import BERT_CRF
    from data_utils import preprocess
    from transformers import BertTokenizer
    from datasets import load_dataset
    from functools import partial

    args = Args(model_name="bert_crf", dataset_name="ner", device=3, output_dir="outputs/bert",
                batch_size=16, num_classes=9, dropout=0.1, pretrained_path="../../../pretrained/bert-base-cased")

    tokenizer = BertTokenizer.from_pretrained("../../../pretrained/bert-base-cased")
    test_dataset = load_dataset("text", data_files="dataset/test.txt", sample_by="paragraph")
    test_dataset = test_dataset.filter(lambda example: example["text"] != "-DOCSTART- -X- -X- O")

    test_dataset = test_dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True, batch_size=None,
                                    remove_columns=["text"])
    test_dataset = test_dataset["train"]

    model = BERT_CRF(args).to(args.device)

    print(evaluate(model, test_dataset, args))
