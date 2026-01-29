from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch

from src.config import default_cfg
from src.features import make_lag_features
from src.splits import add_split_labels
from src.backtesting import generate_fold_anchors
from src.models.temporal_transformer import TemporalTransformerEncoder


def add_perimeter_target_norm(
    df: pd.DataFrame,
    perimeter_col: str,
    y_col: str,
    split_col: str = "split",
    train_split_value: str = "train",
    eps: float = 1e-6,
) -> pd.DataFrame:
    out = df.copy()

    stats = (
        out[out[split_col] == train_split_value]
        .groupby(perimeter_col)[y_col]
        .agg(y_mean="mean", y_std="std")
        .reset_index()
    )
    stats["y_std"] = stats["y_std"].fillna(0.0).clip(lower=eps)

    out = out.merge(stats, on=perimeter_col, how="left")

    g_mean = out[out[split_col] == train_split_value][y_col].mean()
    g_std = out[out[split_col] == train_split_value][y_col].std()
    g_std = float(g_std) if pd.notna(g_std) and g_std > eps else 1.0

    out["y_mean"] = out["y_mean"].fillna(g_mean)
    out["y_std"] = out["y_std"].fillna(g_std).clip(lower=eps)

    out["y_norm"] = (out[y_col] - out["y_mean"]) / out["y_std"]
    return out


@torch.no_grad()
def predict_on_anchors(
    model,
    df_feat: pd.DataFrame,
    ts_col: str,
    perimeter_col: str,
    feature_cols: List[str],
    anchors: List[pd.Timestamp],
    horizons: List[int],
    lookback: int,
    device: str,
) -> pd.DataFrame:
    model.eval()

    df_feat = df_feat.sort_values([perimeter_col, ts_col])
    groups: Dict[str, pd.DataFrame] = {p: g.reset_index(drop=True) for p, g in df_feat.groupby(perimeter_col)}

    rows = []
    for a in anchors:
        for p, g in groups.items():
            idxs = g.index[g[ts_col] == a].to_list()
            if not idxs:
                continue
            t_end = idxs[0]
            if t_end < lookback - 1:
                continue

            x_win = g.loc[t_end - lookback + 1 : t_end, feature_cols]
            if x_win.isna().any().any():
                continue

            xb = torch.from_numpy(x_win.to_numpy(dtype=np.float32)).unsqueeze(0).to(device)
            pred_vec = model(xb).squeeze(0).detach().cpu().numpy()

            for j, h in enumerate(horizons):
                rows.append((a, p, int(h), float(pred_vec[j])))

    return pd.DataFrame(rows, columns=["anchor", "Perimeter", "horizon", "y_pred"])


def main():
    cfg = default_cfg()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lookback = int(getattr(cfg, "lookback_hours", 168))

    ckpt_path = Path("outputs/models/tt_encoder.pt")
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_feature_cols = list(ckpt["feature_cols"])
    horizons = [int(h) for h in ckpt["horizons"]]

    # 1) Load raw
    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.ts_col])
    df[cfg.perimeter_col] = df[cfg.perimeter_col].astype(str).str.strip()

    # 2) Same splits
    df = add_split_labels(
        df, cfg.ts_col,
        cfg.train_start, cfg.train_end,
        cfg.val_start, cfg.val_end,
        cfg.test_start, cfg.test_end
    )

    # 3) Base features
    df_feat = make_lag_features(
        df.copy(), cfg.ts_col, cfg.perimeter_col, cfg.y_col, weather_cols=cfg.weather_cols
    )

    # 4) If checkpoint expects target-normalization-related cols, recreate them
    needs_norm = any(c in ckpt_feature_cols for c in ["y_mean", "y_std", "y_norm"])
    if needs_norm:
        df_feat = add_perimeter_target_norm(
            df=df_feat,
            perimeter_col=cfg.perimeter_col,
            y_col=cfg.y_col,
            split_col="split",
            train_split_value="train",
        )

    # 5) Keep only feature columns that exist (prevents KeyError forever)
    feature_cols = [c for c in ckpt_feature_cols if c in df_feat.columns]
    missing = [c for c in ckpt_feature_cols if c not in df_feat.columns]
    if missing:
        print(f"⚠️ Dropping {len(missing)} missing feature cols not found in df_feat:")
        print(missing[:30], "..." if len(missing) > 30 else "")

    # 6) Numeric + impute (match training logic loosely; enough for inference)
    for c in feature_cols:
        df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    df_feat = df_feat.sort_values([cfg.perimeter_col, cfg.ts_col])
    df_feat[feature_cols] = (
        df_feat.groupby(cfg.perimeter_col, group_keys=False)[feature_cols]
        .apply(lambda g: g.ffill().bfill())
    )
    df_feat[feature_cols] = df_feat[feature_cols].fillna(0.0)

    # 7) Anchors
    anchors = generate_fold_anchors(
        timestamps=df_feat[df_feat["split"] == "test"][cfg.ts_col],
        split_start=cfg.test_start,
        split_end=cfg.test_end,
        lookback_hours=lookback,
        max_horizon_hours=int(max(horizons)),
        stride=cfg.stride
    )
    print(f"Anchors: {len(anchors)}")

    # 8) Model (must match training architecture)
    model = TemporalTransformerEncoder(
        n_features=len(feature_cols),
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        out_horizons=len(horizons),
        max_len=512,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    preds = predict_on_anchors(
        model=model,
        df_feat=df_feat,
        ts_col=cfg.ts_col,
        perimeter_col=cfg.perimeter_col,
        feature_cols=feature_cols,
        anchors=anchors,
        horizons=horizons,
        lookback=lookback,
        device=device,
    )

    out_path = Path("outputs/predictions/preds_tt_encoder_with_perimeter.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)

    print(f"✅ Saved: {out_path}")
    print("Columns:", list(preds.columns))
    print("Rows:", len(preds), "| anchors:", preds["anchor"].nunique(), "| perimeters:", preds["Perimeter"].nunique())


if __name__ == "__main__":
    main()
