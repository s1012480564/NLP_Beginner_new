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


def get_constant_artificial_examples() -> List[Dict]:
    examples = [{"premise": "A man inspects the uniform of a figure in some East Asian country.",
                 "hypothesis": "The man is sleeping.",
                 "relationship": "Contradiction"},
                {"premise": "An older and younger man smiling.",
                 "hypothesis": "Two men are smiling and laughing at the cats playing on the floor.",
                 "relationship": "Neutral"},
                {"premise": "A soccer game with multiple males playing.",
                 "hypothesis": "Some men are playing a sport.",
                 "relationship": "Entailment"}, ]
    return examples


def get_constant_artificial_examples_6shot() -> List[Dict]:
    examples = [{"premise": "A man inspects the uniform of a figure in some East Asian country.",
                 "hypothesis": "The man is sleeping.",
                 "relationship": "Contradiction"},
                {"premise": "An older and younger man smiling.",
                 "hypothesis": "Two men are smiling and laughing at the cats playing on the floor.",
                 "relationship": "Neutral"},
                {"premise": "A soccer game with multiple males playing.",
                 "hypothesis": "Some men are playing a sport.",
                 "relationship": "Entailment"},
                {"premise": "A person on a horse jumps over a broken down airplane.",
                 "hypothesis": "A person is outdoors, on a horse.",
                 "relationship": "Entailment"},
                {"premise": "A person on a horse jumps over a broken down airplane.",
                 "hypothesis": "A person is training his horse for a competition.",
                 "relationship": "Neutral"},
                {"premise": "A person on a horse jumps over a broken down airplane.",
                 "hypothesis": "A person is at a diner, ordering an omelette.",
                 "relationship": "Contradiction"}, ]
    return examples


def _preprocess4knn(examples: Dict[str, List], tokenizer: BartTokenizerFast) -> Dict[str, List]:
    token_encode_dict = tokenizer.batch_encode_plus(list(zip(examples["sentence1"], examples["sentence2"])),
                                                    padding=True, truncation=True)
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
                          remove_columns=["annotator_labels", "captionID", "gold_label", "pairID", "sentence1",
                                          "sentence1_binary_parse", "sentence1_parse", "sentence2",
                                          "sentence2_binary_parse", "sentence2_parse"])
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


def get_knn_examples(neigh_idx: np.ndarray, premise: np.ndarray, hypothesis: np.ndarray, relationship: np.ndarray) -> \
List[List[Dict[str, str]]]:
    return [[{"premise": premise[idx], "hypothesis": hypothesis[idx], "relationship": relationship[idx]} for idx in
             neigh_idx] for neigh_idx in neigh_idx]
