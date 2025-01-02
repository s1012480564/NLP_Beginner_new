from sklearn.neighbors import NearestNeighbors
from transformers import BartTokenizerFast, BartModel
import torch
import pickle
from typing import List, Dict
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import numpy as np
import os


def get_constant_artificial_examples():
    examples = [{"input": "would have a hard time sitting through this one",
                 "output": "Negative"},
                {
                    "input": "A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story .",
                    "output": "Somewhat Negative"},
                {"input": "A series of escapades demonstrating the adage that what is good for the goose",
                 "output": "Neutral"},
                {"input": "good for the goose",
                 "output": "Somewhat Positive"},
                {"input": "This quiet , introspective and entertaining independent is worth seeking .",
                 "output": "Positive"}]
    return examples


def get_variable_artificial_examples_10shot():
    examples = [{"input": "would have a hard time sitting through this one",
                 "output": "Negative"},
                {
                    "input": "A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story .",
                    "output": "Somewhat Negative"},
                {"input": "A series of escapades demonstrating the adage that what is good for the goose",
                 "output": "Neutral"},
                {"input": "good for the goose",
                 "output": "Somewhat Positive"},
                {"input": "This quiet , introspective and entertaining independent is worth seeking .",
                 "output": "Positive"},
                {
                    "input": "A positively thrilling combination of ethnography and all the intrigue , betrayal , deceit and murder",
                    "output": "Positive"},
                {
                    "input": "A positively thrilling combination",
                    "output": "Somewhat Positive"},
                {
                    "input": "is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story",
                    "output": "Neutral"},
                {"input": "but none of which amounts to much of a story",
                 "output": "Somewhat Negative"},
                {"input": "hate it",
                 "output": "Negative"}, ]
    return examples


def _preprocess4knn(examples: Dict[str, List], tokenizer: BartTokenizerFast) -> Dict[str, List]:
    for i in range(len(examples["Phrase"])):
        if examples["Phrase"][i] is None:
            examples["Phrase"][i] = ""
    token_encode_dict = tokenizer.batch_encode_plus(examples["Phrase"], padding=True, truncation=True)
    examples["input_ids"] = token_encode_dict["input_ids"]
    examples["attention_mask"] = token_encode_dict["attention_mask"]
    return examples


def get_knn_example_idx(dataset: Dataset, tokenizer: BartTokenizerFast, model: BartModel,
                        knn_path: str) -> np.ndarray:
    neigh_idx_cached_path = "retriever/neigh_idx.npy"
    if os.path.exists(neigh_idx_cached_path):
        return np.load(neigh_idx_cached_path)

    neigh: NearestNeighbors = pickle.load(open(knn_path, 'rb'))

    dataset = dataset.map(partial(_preprocess4knn, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=["PhraseId", "SentenceId", "Phrase"])
    dataset = dataset.with_format("torch", device=0)
    data_loader = DataLoader(dataset=dataset, batch_size=256)

    score_all = None
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            score = model(batch["input_ids"], batch["attention_mask"]).last_hidden_state[:, 0, :]
            score_all = score if score_all is None else torch.cat((score_all, score), dim=0)

    neigh_idx = neigh.kneighbors(score_all.cpu().numpy(), return_distance=False)

    np.save(neigh_idx_cached_path, neigh_idx)

    return neigh_idx


def get_knn_examples(neigh_idx: np.ndarray, input: np.ndarray, output: np.ndarray) -> \
        List[List[Dict[str, str]]]:
    return [[{"input": input[idx], "output": output[idx]} for idx in neigh_idx] for neigh_idx in neigh_idx]
