import pandas as pd
from pathlib import Path

def load_heart_disease():
    """
    Load the Heart Failure Prediction dataset from Kaggle (fedesoriano).
    Returns
    -------
    X : pandas DataFrame
        Feature matrix.
    y : pandas Series
        Labels (0 = no disease, 1 = disease).
    """
    data_path = Path(__file__).resolve().parent / "data" / "heart.csv"
    df = pd.read_csv(data_path)

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    return X, y
