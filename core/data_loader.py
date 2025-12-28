import pandas as pd
from .config import DATA_PATH, POS_SAMPLES, NEG_SAMPLES, RANDOM_STATE

def load_balanced_subset():
    """
    Loads IMDb dataset from CSV and returns a balanced subset
    for local-resource-friendly training.
    """
    df = pd.read_csv(DATA_PATH)

    # Balanced sampling
    df_pos = df[df["sentiment"] == "positive"].sample(POS_SAMPLES, random_state=RANDOM_STATE)
    df_neg = df[df["sentiment"] == "negative"].sample(NEG_SAMPLES, random_state=RANDOM_STATE)

    # Combine and shuffle
    df_small = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=RANDOM_STATE)

    X = df_small["review"]
    y = df_small["sentiment"]
    return X, y
