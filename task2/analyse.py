import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def compute_and_save_confusion_matrix(labels: np.ndarray, preds: np.ndarray):
    category_names = ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"]
    cm = confusion_matrix(labels, preds)
    cm = cm / np.sum(cm, axis=-1).reshape(-1, 1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("roberta-movie_reviews")
    plt.savefig("outputs/roberta-movie_reviews-confusion_matrix.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    labels = np.load("outputs/movie_reviews-train_labels.npy")
    preds = np.load("outputs/roberta/roberta-movie_reviews-241219-1919-test_preds.npy")
    compute_and_save_confusion_matrix(labels, preds)
