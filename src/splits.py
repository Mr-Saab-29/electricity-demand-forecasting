import pandas as pd

def add_split_labels(
    df: pd.DataFrame,
    ts_col: str,
    train_start: str, train_end: str,
    val_start: str, val_end: str,
    test_start: str, test_end: str
) -> pd.DataFrame:
    out = df.copy()
    out["split"] = "other"

    ts = out[ts_col]
    out.loc[(ts >= train_start) & (ts <= train_end), "split"] = "train"
    out.loc[(ts >= val_start) & (ts <= val_end), "split"] = "val"
    out.loc[(ts >= test_start) & (ts <= test_end), "split"] = "test"

    return out

def filter_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return df[df["split"] == split_name].copy()
