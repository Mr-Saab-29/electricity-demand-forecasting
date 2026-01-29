import pandas as pd
from typing import List

def generate_fold_anchors(
    timestamps: pd.Series,
    split_start: str,
    split_end: str,
    lookback_hours: int,
    max_horizon_hours: int,
    stride: str = "7D",
) -> List[pd.Timestamp]:
    ts = pd.to_datetime(pd.Series(sorted(timestamps.unique())))
    ts_min, ts_max = ts.min(), ts.max()

    start = pd.to_datetime(split_start)
    end = pd.to_datetime(split_end)

    # Ensure we can look back and forward
    start = max(start, ts_min + pd.Timedelta(hours=lookback_hours))
    end = min(end, ts_max - pd.Timedelta(hours=max_horizon_hours))

    anchors = pd.date_range(start=start, end=end, freq=stride)

    # Keep only anchors that exist in the dataset timestamps
    ts_set = set(ts.values)
    anchors = [a for a in anchors if a.to_datetime64() in ts_set]

    return anchors

def make_eval_frame(
    df: pd.DataFrame,
    ts_col: str,
    perimeter_col: str,
    y_col: str,
    anchors: List[pd.Timestamp],
    horizons: List[int],
) -> pd.DataFrame:
    """
    Builds a frame of ground-truth targets for evaluation:
      rows: (anchor, perimeter, horizon, y_true)
    """
    base = df[[ts_col, perimeter_col, y_col]].copy()
    base = base.rename(columns={ts_col: "Timestamp", perimeter_col: "Perimeter", y_col: "y"})

    # index for fast lookup
    base = base.set_index(["Perimeter", "Timestamp"]).sort_index()

    rows = []
    for a in anchors:
        for h in horizons:
            ts_target = a + pd.Timedelta(hours=h)
            rows.append((a, ts_target, h))

    eval_index = pd.DataFrame(rows, columns=["anchor", "target_ts", "horizon"])

    # expand for all perimeters by joining
    perimeters = base.index.get_level_values(0).unique()
    eval_index = eval_index.merge(pd.DataFrame({"Perimeter": perimeters}), how="cross")

    # lookup y_true
    def lookup(row):
        key = (row["Perimeter"], row["target_ts"])
        try:
            return float(base.loc[key, "y"])
        except KeyError:
            return None

    eval_index["y_true"] = eval_index.apply(lookup, axis=1)
    eval_index = eval_index.dropna(subset=["y_true"]).reset_index(drop=True)

    return eval_index
