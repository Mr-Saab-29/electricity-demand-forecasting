from pathlib import Path
import pandas as pd

from src.config import default_cfg
from src.data_loader import load_dataset
from src.splits import add_split_labels
from src.backtesting import generate_fold_anchors, make_eval_frame
from src.features import make_lag_features
from src.models.seasonal_naive import seasonal_naive_predict
from src.models.lgbm import train_lgbm_global, lgbm_predict_anchors
from src.metrics import summarize_metrics

def main():
    cfg = default_cfg()

    out_metrics = Path("outputs/metrics")
    out_preds = Path("outputs/predictions")
    out_metrics.mkdir(parents=True, exist_ok=True)
    out_preds.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = load_dataset(cfg.data_path, cfg.ts_col)

    # 2) Label splits
    df = add_split_labels(
        df, cfg.ts_col,
        cfg.train_start, cfg.train_end,
        cfg.val_start, cfg.val_end,
        cfg.test_start, cfg.test_end
    )

    # 3) Anchors for test
    anchors_test = generate_fold_anchors(
        timestamps=df[df["split"] == "test"][cfg.ts_col],
        split_start=cfg.test_start,
        split_end=cfg.test_end,
        lookback_hours=cfg.lookback_hours,
        max_horizon_hours=cfg.max_horizon,
        stride=cfg.stride
    )
    print(f"Test anchors: {len(anchors_test)} (stride={cfg.stride})")

    # 4) Ground truth evaluation frame
    eval_frame = make_eval_frame(
        df=df,
        ts_col=cfg.ts_col,
        perimeter_col=cfg.perimeter_col,
        y_col=cfg.y_col,
        anchors=anchors_test,
        horizons=cfg.horizons
    )

    # =======================
    # Baseline 1: Seasonal Naive
    # =======================
    pred_naive = seasonal_naive_predict(
        df=df,
        ts_col=cfg.ts_col,
        perimeter_col=cfg.perimeter_col,
        y_col=cfg.y_col,
        anchors=anchors_test,
        horizons=cfg.horizons,
        seasonal_lag_hours=cfg.seasonal_lag_hours
    )

    merged_naive = eval_frame.merge(pred_naive, on=["anchor", "Perimeter", "horizon"], how="inner")
    naive_metrics = summarize_metrics(merged_naive[["horizon", "y_true", "y_pred"]])
    naive_metrics.insert(0, "model", "seasonal_naive")

    merged_naive.to_csv(out_preds / "preds_seasonal_naive.csv", index=False)
    naive_metrics.to_csv(out_metrics / "metrics_seasonal_naive.csv", index=False)

    print("✅ Seasonal naive done")

    # =======================
    # Baseline 2: LightGBM (no weather)
    # =======================
    try:
        df_feat = make_lag_features(
            df.copy(), cfg.ts_col, cfg.perimeter_col, cfg.y_col,
            weather_cols=None
        )

        feature_cols = [c for c in df_feat.columns if c.startswith("lag_")] + [
            "roll_mean_24", "roll_mean_168", "hour", "dow", "month", "is_weekend", "perimeter_id"
        ]

        df_train_feat = df_feat[df_feat["split"].isin(["train", "val"])].dropna(subset=feature_cols + [cfg.y_col]).copy()
        model = train_lgbm_global(df_train_feat, cfg.y_col, feature_cols)

        pred_lgbm = lgbm_predict_anchors(
            df_full=df_feat,
            ts_col=cfg.ts_col,
            perimeter_col=cfg.perimeter_col,
            anchors=anchors_test,
            horizons=cfg.horizons,
            model=model,
            feature_cols=feature_cols
        )

        merged_lgbm = eval_frame.merge(pred_lgbm, on=["anchor", "Perimeter", "horizon"], how="inner")
        lgbm_metrics = summarize_metrics(merged_lgbm[["horizon", "y_true", "y_pred"]])
        lgbm_metrics.insert(0, "model", "lgbm_global")

        merged_lgbm.to_csv(out_preds / "preds_lgbm_global.csv", index=False)
        lgbm_metrics.to_csv(out_metrics / "metrics_lgbm_global.csv", index=False)
        print("✅ LightGBM (no weather) done")

    except Exception as e:
        print("⚠️ LightGBM (no weather) skipped/failed:", e)

    # =======================
    # Baseline 3: LightGBM + Weather
    # =======================

    try:
        df_feat_w = make_lag_features(
        df.copy(), cfg.ts_col, cfg.perimeter_col, cfg.y_col,
        weather_cols=cfg.weather_cols
    )

        # Use only weather cols that exist
        used_weather = [c for c in (cfg.weather_cols or []) if c in df_feat_w.columns]
        feature_cols_w = [c for c in df_feat_w.columns if c.startswith("lag_")] + [
        "roll_mean_24", "roll_mean_168", "hour", "dow", "month", "is_weekend", "perimeter_id"
    ] + used_weather

        if "temperature_2m" in used_weather:
        # only if your make_lag_features adds these
            if "cdh_18" in df_feat_w.columns and "hdh_18" in df_feat_w.columns:
                feature_cols_w += ["cdh_18", "hdh_18"]

        df_train_feat_w = df_feat_w[df_feat_w["split"].isin(["train", "val"])].dropna(subset=feature_cols_w + [cfg.y_col]).copy()
        model_w = train_lgbm_global(df_train_feat_w, cfg.y_col, feature_cols_w)

        pred_lgbm_w = lgbm_predict_anchors(
        df_full=df_feat_w,
        ts_col=cfg.ts_col,
        perimeter_col=cfg.perimeter_col,
        anchors=anchors_test,
        horizons=cfg.horizons,
        model=model_w,
        feature_cols=feature_cols_w
    )

        merged_lgbm_w = eval_frame.merge(pred_lgbm_w, on=["anchor", "Perimeter", "horizon"], how="inner")
        lgbm_metrics_w = summarize_metrics(merged_lgbm_w[["horizon", "y_true", "y_pred"]])
        lgbm_metrics_w.insert(0, "model", "lgbm_weather")

        merged_lgbm_w.to_csv(out_preds / "preds_lgbm_weather.csv", index=False)
        lgbm_metrics_w.to_csv(out_metrics / "metrics_lgbm_weather.csv", index=False)
        print("✅ LightGBM + weather done")

    except Exception as e:
        print("⚠️ LightGBM + weather skipped/failed:", e)

    # =======================
    # Baseline 4: Perimeter-specific LightGBM (WITH weather)
    # =======================
    try:
        from src.models.lgbm_perimeter import train_lgbm_per_perimeter, predict_anchors_perimeter_models

        # df_feat_w and feature_cols_w already built above in the lgbm_weather section
        df_train_feat_w = df_feat_w[df_feat_w["split"].isin(["train", "val"])].dropna(subset=feature_cols_w + [cfg.y_col]).copy()

        per_models = train_lgbm_per_perimeter(
            df_train=df_train_feat_w,
            perimeter_col=cfg.perimeter_col,
            y_col=cfg.y_col,
            feature_cols=feature_cols_w,
        )

        pred_lgbm_per = predict_anchors_perimeter_models(
            df_full=df_feat_w,
            ts_col=cfg.ts_col,
            perimeter_col=cfg.perimeter_col,
            anchors=anchors_test,
            horizons=cfg.horizons,
            models=per_models,
            feature_cols=feature_cols_w,
        )

        merged_per = eval_frame.merge(pred_lgbm_per, on=["anchor", "Perimeter", "horizon"], how="inner")
        per_metrics = summarize_metrics(merged_per[["horizon", "y_true", "y_pred"]])
        per_metrics.insert(0, "model", "lgbm_perimeter_weather")

        merged_per.to_csv(out_preds / "preds_lgbm_perimeter_weather.csv", index=False)
        per_metrics.to_csv(out_metrics / "metrics_lgbm_perimeter_weather.csv", index=False)
        print("✅ Perimeter-specific LightGBM + weather done")

    except Exception as e:
        print("⚠️ Perimeter-specific LightGBM skipped/failed:", e)
    
    # =======================
    # Baseline 5: Temporal Transformer Encoder (precomputed preds)
    # =======================
    try:
        tt_path = out_preds / "preds_tt_encoder.csv"  # ✅ must be saved here by your transformer script

        if not tt_path.exists():
            raise FileNotFoundError(
                f"Missing transformer preds file: {tt_path}. "
                f"Make sure your transformer training script writes to outputs/predictions/preds_tt_encoder.csv"
            )

        pred_tt = pd.read_csv(tt_path)
        pred_tt["anchor"] = pd.to_datetime(pred_tt["anchor"])
        pred_tt["horizon"] = pred_tt["horizon"].astype(int)

        # Normalize Perimeter just in case
        pred_tt["Perimeter"] = pred_tt["Perimeter"].astype(str).str.strip()
        eval_frame["Perimeter"] = eval_frame["Perimeter"].astype(str).str.strip()

        merged_tt = eval_frame.merge(pred_tt, on=["anchor", "Perimeter", "horizon"], how="inner")
        if merged_tt.empty:
            raise ValueError(
                "Transformer merge produced 0 rows. Check that your transformer preds use:\n"
                "- the same Perimeter names as eval_frame\n"
                "- the same anchors_test timestamps\n"
                "- the same horizons list\n"
            )

        tt_metrics = summarize_metrics(merged_tt[["horizon", "y_true", "y_pred"]])
        tt_metrics.insert(0, "model", "tt_encoder_weather")

        merged_tt.to_csv(out_preds / "preds_tt_encoder_merged.csv", index=False)
        tt_metrics.to_csv(out_metrics / "metrics_tt_encoder_weather.csv", index=False)
        print("✅ Temporal Transformer Encoder metrics done")

    except Exception as e:
        print("⚠️ Temporal Transformer Encoder skipped/failed:", e)



    # Combine metrics
    all_metrics = [pd.read_csv(out_metrics / "metrics_seasonal_naive.csv")]
    for fname in ["metrics_lgbm_global.csv", "metrics_lgbm_weather.csv", "metrics_lgbm_perimeter_weather.csv", "metrics_tt_encoder_weather.csv", "metrics_tt_encoder.csv"]:
        p = out_metrics / fname
        if p.exists():
            all_metrics.append(pd.read_csv(p))

    pd.concat(all_metrics, ignore_index=True).to_csv(out_metrics / "metrics_all.csv", index=False)
    print("✅ Saved combined metrics -> outputs/metrics/metrics_all.csv")

if __name__ == "__main__":
    main()
