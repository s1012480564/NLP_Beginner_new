import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("dataset/test.csv")
    phrase_id = df["PhraseId"]
    preds = np.load("outputs/KATE-llama-3.2-1b-instruct-movie_reviews-sentiment-test_preds.npy")
    df = pd.DataFrame({"PhraseId": phrase_id, "Sentiment": preds})
    df.to_csv("submission/submission_KATE-llama-3.2-1b-instruct-sentiment.csv", index=False)
