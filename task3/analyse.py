import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def compute_and_save_confusion_matrix(labels: np.ndarray, preds: np.ndarray):
    category_names = ["contradiction", "neutral", "entailment"]
    cm = confusion_matrix(labels, preds)
    cm = cm / np.sum(cm, axis=-1).reshape(-1, 1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("self_exlain_roberta-snli")
    plt.savefig("outputs/self_exlain_roberta-snli-confusion_matrix.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    labels = np.load("outputs/snli-test_labels.npy")
    preds = np.load("outputs/roberta/self_explain_roberta-snli-241221-2309-test_preds.npy")
    compute_and_save_confusion_matrix(labels, preds)
