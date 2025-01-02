# 阅读 KATE 源码得到的建议：取 bart-large [CLS] 处 encode 输出。而不取平均，或用 sentence transformer
# 距离计算采用 euclidian 即 sklearn KNN 默认的 minkowski
# 阅读原文，取最近的，而不取最远的
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import BartTokenizerFast, BartModel
from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import pickle


def _preprocess4knn(examples: Dict[str, List], tokenizer: BartTokenizerFast) -> Dict[str, List]:
    token_encode_dict = tokenizer.batch_encode_plus(list(zip(examples["sentence1"], examples["sentence2"])),
                                                    padding=True, truncation=True)
    examples["input_ids"] = token_encode_dict["input_ids"]
    examples["attention_mask"] = token_encode_dict["attention_mask"]
    return examples


if __name__ == '__main__':
    batch_size = 256
    device = 0
    pretrained_path = "../../../../pretrained/bart-large"

    tokenizer = BartTokenizerFast.from_pretrained(pretrained_path)

    dataset = load_dataset("json", data_files="../dataset/snli_1.0_train.jsonl")
    dataset = dataset["train"]
    dataset = dataset.filter(lambda example: example["gold_label"] in ["contradiction", "neutral", "entailment"])
    dataset = dataset.map(partial(_preprocess4knn, tokenizer=tokenizer), batched=True, batch_size=None,
                          remove_columns=["annotator_labels", "captionID", "gold_label", "pairID", "sentence1",
                                          "sentence1_binary_parse", "sentence1_parse", "sentence2",
                                          "sentence2_binary_parse", "sentence2_parse"])
    dataset = dataset.with_format("torch", device=device)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size)

    score_all = None
    model = BartModel.from_pretrained(pretrained_path).to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            score = model(batch["input_ids"], batch["attention_mask"]).last_hidden_state[:, 0, :]
            score_all = score if score_all is None else torch.cat((score_all, score), dim=0)

    neigh = NearestNeighbors(n_neighbors=3, algorithm="ball_tree", n_jobs=-1)
    neigh.fit(score_all.cpu().numpy())
    pickle.dump(neigh, open("knn_fit.dat", 'wb'))
