from typing import List

import numpy as np
import pandas as pd


def load_data(filenames: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files with telemetry runs.
    """
    dfs = [pd.read_csv(f) for f in filenames]
    return pd.concat(dfs, ignore_index=True)


def feature_engineering(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Simple feature engineering:
    - rolling mean and std for each channel over a given window size.
    Returns a new dataframe with features and label.
    """
    df_feat = pd.DataFrame()

    channels = ["Pc", "N_pump", "T_in", "Vib"]

    for col in channels:
        df_feat[f"{col}_mean"] = df[col].rolling(window).mean()
        df_feat[f"{col}_std"] = df[col].rolling(window).std()

    df_feat["label"] = df["label"]

    # drop first rows where rolling window is not full
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat
