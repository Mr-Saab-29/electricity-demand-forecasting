import numpy as np
import pandas as pd

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def summarize_metrics(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    df_pred must have columns: horizon, y_true, y_pred, (optional) Perimeter
    Returns metrics per horizon + overall.
    """
    out = []
    for h, g in df_pred.groupby("horizon"):
        out.append({
            "horizon": int(h),
            "mae": mae(g["y_true"], g["y_pred"]),
            "rmse": rmse(g["y_true"], g["y_pred"]),
            "mape": mape(g["y_true"], g["y_pred"]),
            "n": len(g),
        })

    overall = {
        "horizon": "overall",
        "mae": mae(df_pred["y_true"], df_pred["y_pred"]),
        "rmse": rmse(df_pred["y_true"], df_pred["y_pred"]),
        "mape": mape(df_pred["y_true"], df_pred["y_pred"]),
        "n": len(df_pred),
    }
    out.append(overall)
    return pd.DataFrame(out)
