import pandas as pd
from typing import List

def seasonal_naive_predict(
    df: pd.DataFrame,
    ts_col: str,
    perimeter_col: str,
    y_col: str,
    anchors: List[pd.Timestamp],
    horizons: List[int],
    seasonal_lag_hours: int = 168
) -> pd.DataFrame:
    """
    y_pred(anchor, h) = y_true at (anchor + h - seasonal_lag_hours)
    """
    base = df[[ts_col, perimeter_col, y_col]].copy()
    base = base.rename(columns={ts_col: "Timestamp", perimeter_col: "Perimeter", y_col: "y"})
    base = base.set_index(["Perimeter", "Timestamp"]).sort_index()

    rows = []
    perimeters = base.index.get_level_values(0).unique()

    for a in anchors:
        for h in horizons:
            pred_ts = a + pd.Timedelta(hours=h - seasonal_lag_hours)
            for p in perimeters:
                try:
                    yhat = float(base.loc[(p, pred_ts), "y"])
                    rows.append((a, p, h, yhat))
                except KeyError:
                    continue

    return pd.DataFrame(rows, columns=["anchor", "Perimeter", "horizon", "y_pred"])
