import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datasets import ClassLabel
import argparse
from config_utils import Args
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler("logs/inference.log"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name", default="KATE", type=str)
    parser.add_argument("--model_name", default="llama-3.2-1b-instruct", type=str)
    parser.add_argument("--dataset_name", default="snli", type=str)
    parser.add_argument("--template_name", default="NLI", type=str)
    parser.add_argument("--instruction", default="", type=str, help="e.g. \"instruction\" or empty str \"\" ")
    parser_args = parser.parse_args()
    args = Args(**vars(parser_args))

    class_label = ClassLabel(names=["Contradiction", "Neutral", "Entailment"])
    labels = np.load("outputs/snli-test_labels.npy")
    preds = np.load(
        f"outputs/{args.method_name}-{args.model_name}-{args.dataset_name}-{args.template_name}{"-" + args.instruction if args.instruction else ""}-test_preds.npy")

    acc = round(metrics.accuracy_score(labels, preds), 4)
    macro_f1 = round(metrics.f1_score(labels, preds, labels=range(class_label.num_classes), average="macro"), 4)

    logger.info(
        f"{args.method_name}-{args.model_name}-{args.dataset_name}-{args.template_name}{"-" + args.instruction if args.instruction else ""}:")
    logger.info(f"acc: {acc}, f1: {macro_f1}\n")

    cm = confusion_matrix(labels, preds)
    cm = cm / np.sum(cm, axis=-1).reshape(-1, 1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_label.names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(
        f"{args.method_name}-{args.model_name}-{args.dataset_name}-{args.template_name}{"-" + args.instruction if args.instruction else ""}")
    plt.savefig(
        f"outputs/{args.method_name}-{args.model_name}-{args.dataset_name}-{args.template_name}{"-" + args.instruction if args.instruction else ""}-confusion_matrix.png",
        dpi=300, bbox_inches='tight')
