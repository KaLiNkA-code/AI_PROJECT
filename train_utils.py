from typing import Dict, Tuple

import numpy as np  # np
import pandas as pd


def clean_df(df: pd.DataFrame):
    # Drop NaN, replace columns e.t.c
    df = df[(df["budget"] > 0) & (df["popularity"] > 0)]
    df.dropna(subset=["budget", "popularity"], inplace=True)
    return df


def train_val_test_split(df: pd.DataFrame, train_pct: float = 0.6, val_pct: float = 0.2, test_pct: float = 0.2):
    N = len(df)

    if train_pct + val_pct + test_pct != 1.0:
        train_pct = train_pct / (train_pct + val_pct + test_pct)
        val_pct = val_pct / (train_pct + val_pct + test_pct)
        test_pct = test_pct / (train_pct + val_pct + test_pct)

    len_train = int(N * train_pct)
    len_val = int(N * val_pct)
    print(len_train, len_val)
    df.sample()
    train_df, val_df, test_df = df[:len_train], df[len_train:len_val], df[len_val:]
    return train_df, val_df, test_df


def preprocess_df(
    df: pd.DataFrame, columns: tuple = ("budget", "popularity", "revenue"), stats=None
) -> Tuple[pd.DataFrame, Dict]:
    #     Tuple[pd.DataFrame, Dict] tuple(pd.DataFrame, dict)
    # Union[Dict, List] ->  dict | list
    # Select columns

    df = df[list(columns)]
    if stats is None:
        stats = {
            "mean": df.mean(),
            "std": df.std(),
        }
        # Считаем распределния и нормализуем по данным из DF
        # Standard Scaling
        #          .loc[row_indexer,col_indexer] = value instead
        try:
            for i in columns:
                df[i] = (df[i] - stats["mean"][i]) / stats["std"][i]
                #  df.sample(10)
        except KeyError:
            return "Column not found"
        except BaseException as Err:
            return f"TraceBack: {Err}"
    else:
        stats = {
            "mean": df.mean(),
            "meansquare": df.std(),
        }
    return df, stats


# MSE loss function
def mse_loss(val_predict: np.ndarray, val_true: np.ndarray):
    squared_error = (val_predict - val_true) ** 2
    sum_squared_error = np.sum(squared_error)
    loss = sum_squared_error / val_true.size
    return loss
