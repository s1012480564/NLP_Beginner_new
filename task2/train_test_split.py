import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv("dataset/train.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("dataset/train_split.csv", index=False)
    val_df.to_csv("dataset/val_split.csv", index=False)
