import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from config_utils import Args
from models import CNN, LSTM, RoBERTa
import pickle
from safetensors.torch import load_file
from torch.utils.data import DataLoader
import pandas as pd


def _predict4submission(test_dataset: Dataset, args: Args) -> None:
    model, tokenizer = None, None
    if args.model_name == "cnn" or args.model_name == "lstm":
        tokenizer = pickle.load(open("glove_pre/movie_reviews_tokenizer.dat", 'rb'))
    else:
        tokenizer = AutoTokenizer.from_pretrained("../../../pretrained/roberta-base")

    if args.model_name == "cnn" or args.model_name == "lstm":
        test_dataset = test_dataset.map(
            lambda examples: {
                "input_ids": tokenizer.batch_encode_plus(examples["Phrase"], padding=True, truncation=True)},
            batched=True,
            batch_size=None, remove_columns=["Phrase", "SentenceId"])
    else:
        test_dataset = test_dataset.map(
            lambda examples: tokenizer.batch_encode_plus(examples["Phrase"], padding=True, truncation=True),
            batched=True,
            batch_size=None, remove_columns=["Phrase", "SentenceId"])

    match args.model_name:
        case "cnn":
            model = CNN(args).to(args.device)
        case "lstm":
            model = LSTM(args).to(args.device)
        case "roberta":
            model = RoBERTa(args).to(args.device)

    state_dict = load_file("outputs/" + args.model_name + "/model.safetensors")
    model.load_state_dict(state_dict)

    test_dataset = test_dataset.with_format("torch", device=args.device)
    data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    preds_all = None
    model.eval()
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(data_loader)):
            logits = model(**batch)["logits"]
            preds = logits.argmax(dim=-1)
            preds_all = torch.cat([preds_all, preds]) if preds_all is not None else preds
    preds_all = preds_all.cpu().numpy()
    phrase_id = test_dataset["PhraseId"].cpu().numpy()
    df = pd.DataFrame({"PhraseId": phrase_id, "Sentiment": preds_all})
    df.to_csv("submission/submission_" + args.model_name + ".csv", index=False)


if __name__ == '__main__':
    args = Args(device=0, batch_size=256, num_classes=5, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/roberta-base")

    test_dataset = load_dataset("csv", data_files="dataset/test.csv")
    # 框架所限，得随便填个 label
    test_dataset = test_dataset.map(
        lambda example: {"Phrase": example["Phrase"], "labels": 0} if example["Phrase"] is not None else {"Phrase": "",
                                                                                                          "labels": 0})

    test_dataset = test_dataset["train"]

    args.model_name = "cnn"
    _predict4submission(test_dataset, args)

    args.model_name = "lstm"
    _predict4submission(test_dataset, args)

    args.model_name = "roberta"
    _predict4submission(test_dataset, args)
