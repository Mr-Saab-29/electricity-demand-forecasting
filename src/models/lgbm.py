import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def train_lgbm_global(df_train: pd.DataFrame, y_col: str, feature_cols: List[str]):
    if lgb is None:
        raise ImportError("lightgbm is not installed. Install it or skip LightGBM baseline.")

    X = df_train[feature_cols]
    y = df_train[y_col]

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    return model


def lgbm_predict_anchors(
    df_full: pd.DataFrame,
    ts_col: str,
    perimeter_col: str,
    anchors: List[pd.Timestamp],
    horizons: List[int],
    model,
    feature_cols: List[str]
) -> pd.DataFrame:
    rows = []
    df_idx = df_full.set_index([perimeter_col, ts_col]).sort_index()
    perimeters = df_full[perimeter_col].unique()

    for a in anchors:
        for h in horizons:
            t = a + pd.Timedelta(hours=h)
            for p in perimeters:
                try:
                    row = df_idx.loc[(p, t)]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    x = row[feature_cols].to_frame().T
                    x = x.apply(pd.to_numeric, errors="coerce")
                    if x.isna().any(axis=None):
                        continue
                    yhat = float(model.predict(x)[0])
                    rows.append((a, p, h, yhat))
                except KeyError:
                    continue

    return pd.DataFrame(rows, columns=["anchor", "Perimeter", "horizon", "y_pred"])
