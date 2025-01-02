import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def compute_and_save_confusion_matrix(labels: np.ndarray, preds: np.ndarray):
    category_names = ["B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER", "O"]
    cm = confusion_matrix(labels, preds)
    cm = cm / np.sum(cm, axis=-1).reshape(-1, 1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("bert_crf-ner")
    plt.savefig("outputs/bert_crf-ner-confusion_matrix.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    labels = np.load("outputs/ner-test_labels.npy")
    preds = np.load("outputs/bert/bert_crf-ner-241224-2054-test_preds.npy")
    compute_and_save_confusion_matrix(labels, preds)
