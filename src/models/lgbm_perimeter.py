import pandas as pd
from typing import Dict, List, Any

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def train_lgbm_per_perimeter(
    df_train: pd.DataFrame,
    perimeter_col: str,
    y_col: str,
    feature_cols: List[str],
) -> Dict[str, Any]:
    if lgb is None:
        raise ImportError("lightgbm is not installed.")

    models = {}
    for p, g in df_train.groupby(perimeter_col):
        X = g[feature_cols]
        y = g[y_col]

        model = lgb.LGBMRegressor(
            n_estimators=4000,
            learning_rate=0.02,
            num_leaves=128,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        # simple fit; you can add early stopping if you want
        model.fit(X, y)
        models[p] = model

    return models


def predict_anchors_perimeter_models(
    df_full: pd.DataFrame,
    ts_col: str,
    perimeter_col: str,
    anchors,
    horizons,
    models: Dict[str, Any],
    feature_cols: List[str],
) -> pd.DataFrame:
    rows = []
    df_idx = df_full.set_index([perimeter_col, ts_col]).sort_index()

    for a in anchors:
        for h in horizons:
            t = a + pd.Timedelta(hours=h)
            for p, model in models.items():
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
