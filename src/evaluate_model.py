from pathlib import Path
import pandas as pd

from src.config import default_cfg
from src.backtesting import make_eval_frame
from src.metrics import summarize_metrics


def main():
    cfg = default_cfg()

    preds_path = Path("outputs/predictions/preds_tt_encoder.csv")
    assert preds_path.exists(), "Predictions file not found"

    preds = pd.read_csv(preds_path)

    # Basic schema check before parsing
    required_cols = {"anchor", "Perimeter", "horizon", "y_pred"}
    missing = required_cols - set(preds.columns)
    if missing:
        raise ValueError(
            f"Predictions missing columns: {missing}. Found: {list(preds.columns)}. "
            f"Check {preds_path} for a correct header and delimiter."
        )

    # Normalize types to avoid anchor + Timedelta errors
    preds["anchor"] = pd.to_datetime(preds["anchor"], errors="coerce")
    preds["horizon"] = pd.to_numeric(preds["horizon"], errors="coerce").astype("Int64")
    preds["Perimeter"] = preds["Perimeter"].astype(str).str.strip()

    # Drop invalid rows early
    preds = preds.dropna(subset=["anchor", "horizon", "Perimeter", "y_pred"]).copy()
    preds["horizon"] = preds["horizon"].astype(int)

    # Rebuild eval frame (ground truth)
    df = pd.read_csv(cfg.data_path, parse_dates=[cfg.ts_col])

    eval_frame = make_eval_frame(
        df=df,
        ts_col=cfg.ts_col,
        perimeter_col=cfg.perimeter_col,
        y_col=cfg.y_col,
        anchors=preds["anchor"].unique().tolist(),
        horizons=sorted(preds["horizon"].unique().tolist()),
    )

    # Align keys (defensive)
    eval_frame["Perimeter"] = eval_frame["Perimeter"].astype(str).str.strip()
    eval_frame["anchor"] = pd.to_datetime(eval_frame["anchor"], errors="coerce")

    merged = eval_frame.merge(
        preds,
        on=["anchor", "Perimeter", "horizon"],
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            "Merge produced 0 rows. Check that preds use the same anchors, horizons, "
            "and Perimeter names as the eval frame."
        )

    metrics = summarize_metrics(merged[["horizon", "y_true", "y_pred"]])
    metrics.insert(0, "model", "tt_encoder")

    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics_tt_encoder.csv", index=False)

    print("✅ Metrics computed without retraining")
    print(metrics)


if __name__ == "__main__":
    main()
