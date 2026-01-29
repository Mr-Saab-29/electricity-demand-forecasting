from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import default_cfg
from src.features import make_lag_features
from src.models.temporal_transformer import TemporalTransformerEncoder
from src.splits import add_split_labels
from src.backtesting import generate_fold_anchors

import time
from tqdm import tqdm
import psutil

# ------------------------
# Dataset (robust: skips invalid windows)
# ------------------------
class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ts_col: str,
        perimeter_col: str,
        y_col: str,
        feature_cols: List[str],
        horizons: List[int],
        lookback: int,
        split_values: List[str],
    ):
        self.ts_col = ts_col
        self.perimeter_col = perimeter_col
        self.y_col = y_col
        self.feature_cols = feature_cols
        self.horizons = horizons
        self.lookback = lookback

        d = df[df["split"].isin(split_values)].copy()
        d = d.sort_values([perimeter_col, ts_col])

        self.groups: Dict[str, pd.DataFrame] = {
            p: g.reset_index(drop=True) for p, g in d.groupby(perimeter_col)
        }
        self.items: List[Tuple[str, int]] = []

        max_h = int(max(horizons))

        for p, g in self.groups.items():
            if len(g) < (lookback + max_h + 1):
                continue

            for t_end in range(lookback - 1, len(g) - max_h):
                x_win = g.loc[t_end - lookback + 1 : t_end, feature_cols]
                if x_win.isna().any().any():
                    continue

                idxs = [t_end + int(h) for h in horizons]
                y_vec = g.loc[idxs, y_col]
                if y_vec.isna().any():
                    continue

                self.items.append((p, t_end))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, t_end = self.items[idx]
        g = self.groups[p]

        x_win = g.loc[t_end - self.lookback + 1 : t_end, self.feature_cols].to_numpy(dtype=np.float32)
        y_vec = g.loc[[t_end + int(h) for h in self.horizons], self.y_col].to_numpy(dtype=np.float32)

        return torch.from_numpy(x_win), torch.from_numpy(y_vec)


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optim, device, epoch: int, total_epochs: int):
    model.train()
    loss_fn = nn.SmoothL1Loss()
    total = 0.0
    n = 0

    pbar = tqdm(
        loader,
        desc=f"Train {epoch:02d}/{total_epochs}",
        leave=False,
        ncols=110,
        dynamic_ncols=True,
    )

    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device)

        optim.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        total += loss.item() * xb.size(0)
        n += xb.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total/max(n,1)):.4f}")

    return total / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device, epoch: int, total_epochs: int):
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    total = 0.0
    n = 0

    pbar = tqdm(
        loader,
        desc=f"Val   {epoch:02d}/{total_epochs}",
        leave=False,
        ncols=110,
        dynamic_ncols=True,
    )

    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)

        total += loss.item() * xb.size(0)
        n += xb.size(0)

        pbar.set_postfix(val_loss=f"{loss.item():.4f}", avg=f"{(total/max(n,1)):.4f}")

    return total / max(n, 1)



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
    groups = {p: g.reset_index(drop=True) for p, g in df_feat.groupby(perimeter_col)}

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

def add_perimeter_target_norm(
    df: pd.DataFrame,
    perimeter_col: str,
    y_col: str,
    split_col: str = "split",
    train_split_value: str = "train",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Adds train-only per-perimeter mean/std, plus y_norm.
    No leakage: stats are computed ONLY on train split.
    """
    out = df.copy()

    stats = (
        out[out[split_col] == train_split_value]
        .groupby(perimeter_col)[y_col]
        .agg(y_mean="mean", y_std="std")
        .reset_index()
    )

    # If some perimeters have very low std or std NaN (rare), stabilize.
    stats["y_std"] = stats["y_std"].fillna(0.0)
    stats["y_std"] = stats["y_std"].clip(lower=eps)

    out = out.merge(stats, on=perimeter_col, how="left")

    # If a perimeter somehow has no train rows, fallback to global train stats
    g_mean = out[out[split_col] == train_split_value][y_col].mean()
    g_std = out[out[split_col] == train_split_value][y_col].std()
    g_std = float(g_std) if pd.notna(g_std) and g_std > eps else 1.0

    out["y_mean"] = out["y_mean"].fillna(g_mean)
    out["y_std"] = out["y_std"].fillna(g_std).clip(lower=eps)

    out["y_norm"] = (out[y_col] - out["y_mean"]) / out["y_std"]
    return out


def denorm_preds_perimeter(
    preds: pd.DataFrame,
    df_with_stats: pd.DataFrame,
    perimeter_col: str,
) -> pd.DataFrame:
    """
    preds: columns [anchor, Perimeter, horizon, y_pred] where y_pred is normalized.
    Uses per-perimeter y_mean/y_std stored in df_with_stats.
    Returns same preds with y_pred replaced to original scale.
    """
    stats = (
        df_with_stats[[perimeter_col, "y_mean", "y_std"]]
        .drop_duplicates(subset=[perimeter_col])
        .copy()
    )

    out = preds.merge(
        stats,
        left_on="Perimeter",
        right_on=perimeter_col,
        how="left",
    )

    out["y_pred"] = out["y_pred"] * out["y_std"] + out["y_mean"]
    cols_to_drop = ["y_mean", "y_std"]
    if perimeter_col != "Perimeter":
        cols_to_drop.append(perimeter_col)
    out = out.drop(columns=cols_to_drop)
    return out


def main():
    seed_everything(42)
    cfg = default_cfg()

    # For first run (CPU-friendly), uncomment next line:
    # horizons = list(range(1, 25))
    horizons = cfg.horizons
    lookback = int(getattr(cfg, "lookback_hours", 168))

    TOTAL_EPOCHS = 30
    start_time = time.time()

    # 1) Load
    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.ts_col])
    df[cfg.perimeter_col] = df[cfg.perimeter_col].astype(str).str.strip()

    # 2) SAME split labeling as run_backtest.py
    df = add_split_labels(
        df, cfg.ts_col,
        cfg.train_start, cfg.train_end,
        cfg.val_start, cfg.val_end,
        cfg.test_start, cfg.test_end
    )

    # 3) Features (weather-aware)
    df_feat = make_lag_features(
        df.copy(), cfg.ts_col, cfg.perimeter_col, cfg.y_col, weather_cols=cfg.weather_cols
    )

    # 4) Keep rows with y
    df_feat = df_feat.dropna(subset=[cfg.y_col]).copy()

    #  5) Add per-perimeter target normalization (train-only stats)
    df_feat = add_perimeter_target_norm(
        df=df_feat,
        perimeter_col=cfg.perimeter_col,
        y_col=cfg.y_col,
        split_col="split",
        train_split_value="train",
    )

    df_feat["y_norm_lag1"] = df_feat.groupby(cfg.perimeter_col)["y_norm"].shift(1)

    # 6) Transformer feature cols: remove lag_*, roll_* engineered
    drop_cols = {cfg.y_col, cfg.perimeter_col, cfg.ts_col, "split", "y_norm"}  # drop y_norm raw!
    feature_cols = []
    for c in df_feat.columns:
        if c in drop_cols or c in {"Date", "Hour"}:
            continue
        if c.startswith("lag_"):
            continue
        if c.startswith("roll_"):
            continue
        feature_cols.append(c)

    # Ensure we include lagged normalized target
    if "y_norm_lag1" not in feature_cols:
        feature_cols.append("y_norm_lag1")

    # Target for training
    y_train_col = "y_norm"

    for c in feature_cols + [y_train_col]:
        df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")

    # Keep rows where target exists (normalized target)
    df_feat = df_feat.dropna(subset=[y_train_col]).copy()


    # ---------------------------
    # NEW: diagnose missingness in TRAIN, drop toxic cols, then impute
    # ---------------------------
    train_mask = df_feat["split"] == "train"
    miss = df_feat.loc[train_mask, feature_cols].isna().mean().sort_values(ascending=False)
    print("\nTop missing feature cols (train split):")
    print(miss.head(20))

    # Drop columns that are too missing in train
    drop_bad = miss[miss > 0.10].index.tolist()  # start with 10%
    if drop_bad:
        print(f"\nDropping {len(drop_bad)} feature cols with >10% missing in train.")
        feature_cols = [c for c in feature_cols if c not in drop_bad]

    # Impute per perimeter (ffill/bfill), then median/0
    df_feat = df_feat.sort_values([cfg.perimeter_col, cfg.ts_col])

    df_feat[feature_cols] = (
        df_feat.groupby(cfg.perimeter_col, group_keys=False)[feature_cols]
        .apply(lambda g: g.ffill().bfill())
    )

    train_medians = df_feat.loc[train_mask, feature_cols].median(numeric_only=True)
    df_feat[feature_cols] = df_feat[feature_cols].fillna(train_medians).fillna(0.0)

    miss2 = df_feat.loc[train_mask, feature_cols].isna().mean().sort_values(ascending=False)
    print("\nTop missing AFTER imputation (train split):")
    print(miss2.head(10))

    # Diagnostics
    print("\nRows by split:",
          "train=", int((df_feat["split"] == "train").sum()),
          "val=", int((df_feat["split"] == "val").sum()),
          "test=", int((df_feat["split"] == "test").sum()))
    print("lookback=", lookback, "max_horizon=", int(max(horizons)), "n_features=", len(feature_cols))


    # 7) Dataset
    train_ds = SequenceDataset(df_feat, cfg.ts_col, cfg.perimeter_col, y_train_col, feature_cols, horizons, lookback, ["train"])
    val_ds   = SequenceDataset(df_feat, cfg.ts_col, cfg.perimeter_col, y_train_col, feature_cols, horizons, lookback, ["val"])


    print("Samples:", "train=", len(train_ds), "val=", len(val_ds))
    if len(train_ds) == 0:
        raise ValueError(
            "Train dataset has 0 samples even after imputation.\n"
            "Likely: lookback/max_horizon too large for contiguous data OR split has gaps.\n"
            "Try first-run fast mode: horizons=1..24 and lookback=72."
        )

    # 8) DataLoaders
    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    # 9) Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = TemporalTransformerEncoder(
        n_features=len(feature_cols),
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        out_horizons=len(horizons),
        max_len=512
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # 10) Train
    best = float("inf")
    patience = 5
    bad = 0

    Path("outputs/models").mkdir(parents=True, exist_ok=True)

    print("FINAL CONFIG")
    print("lookback:", lookback)
    print("horizons:", horizons)
    print("n_features:", len(feature_cols))
    print("features:", feature_cols[:10], "...")


    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start = time.time()

        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optim=optim,
            device=device,
            epoch=epoch,
            total_epochs=TOTAL_EPOCHS,
        )

        va = eval_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            total_epochs=TOTAL_EPOCHS,
        ) if len(val_ds) > 0 else tr

        epoch_time = time.time() - epoch_start
        eta = (TOTAL_EPOCHS - epoch) * epoch_time

        # CPU/RAM
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        # GPU mem proof
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            gpu_line = f"GPUmem={alloc:.0f}/{reserved:.0f}MB"
        else:
            gpu_line = "GPUmem=n/a"

        print(
            f"epoch={epoch:02d}/{TOTAL_EPOCHS} "
            f"train={tr:.5f} val={va:.5f} "
            f"epoch_time={epoch_time/60:.1f}m ETA={eta/60:.1f}m "
            f"CPU={cpu:.0f}% RAM={ram:.0f}% {gpu_line}"
        )

        if va < best - 1e-4:
            best = va
            bad = 0
            torch.save(
                {"state_dict": model.state_dict(), "feature_cols": feature_cols, "horizons": horizons},
                "outputs/models/tt_encoder.pt"
            )
            print("  ✅ saved best checkpoint")
        else:
            bad += 1
            print(f"  ⚠️ no improvement (bad={bad}/{patience})")
            if bad >= patience:
                print("Early stopping.")
                break


    # 11) Load best
    ckpt = torch.load("outputs/models/tt_encoder.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    feature_cols = ckpt["feature_cols"]
    horizons = ckpt["horizons"]

    # 9) Anchors (same as run_backtest.py)
    anchors = generate_fold_anchors(
        timestamps=df_feat[df_feat["split"] == "test"][cfg.ts_col],
        split_start=cfg.test_start,
        split_end=cfg.test_end,
        lookback_hours=lookback,
        max_horizon_hours=int(max(horizons)),
        stride=cfg.stride
    )
    print(f"Predicting on anchors: {len(anchors)}")

    preds = predict_on_anchors(
        model=model,
        df_feat=df_feat,
        ts_col=cfg.ts_col,
        perimeter_col=cfg.perimeter_col,
        feature_cols=feature_cols,
        anchors=anchors,
        horizons=horizons,
        lookback=lookback,
        device=device
    )

    # preds are in normalized scale -> convert back to original MWh
    preds = denorm_preds_perimeter(
        preds=preds,
        df_with_stats=df_feat,
        perimeter_col=cfg.perimeter_col,
    )


    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    preds.to_csv("outputs/predictions/preds_tt_encoder.csv", index=False)
    print("✅ Saved outputs/predictions/preds_tt_encoder.csv")

    from src.backtesting import make_eval_frame

    eval_frame = make_eval_frame(
        df=df,
        ts_col=cfg.ts_col,
        perimeter_col=cfg.perimeter_col,
        y_col=cfg.y_col,
        anchors=preds["anchor"].unique().tolist(),
        horizons=sorted(preds["horizon"].unique().tolist()),
    )

    preds["Perimeter"] = preds["Perimeter"].astype(str).str.strip()
    eval_frame["Perimeter"] = eval_frame["Perimeter"].astype(str).str.strip()

    merged = eval_frame.merge(
        preds,
        on=["anchor", "Perimeter", "horizon"],
        how="inner",
    )

    merged.to_csv("outputs/predictions/preds_tt_encoder_merged.csv", index=False)
    print("✅ Saved outputs/predictions/preds_tt_encoder_merged.csv")

if __name__ == "__main__":

    main()
