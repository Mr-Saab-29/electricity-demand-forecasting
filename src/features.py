import pandas as pd
from typing import List

def add_calendar_features(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = out[ts_col]

    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    return out

def add_perimeter_id(df: pd.DataFrame, perimeter_col: str) -> pd.DataFrame:
    out = df.copy()
    codes, uniques = pd.factorize(out[perimeter_col], sort=True)
    out["perimeter_id"] = codes.astype(int)
    return out

def make_lag_features(
    df: pd.DataFrame,
    ts_col: str,
    perimeter_col: str,
    y_col: str,
    weather_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Creates lag + rolling + calendar + perimeter_id + weather-aware features.
    Assumes df already contains weather columns if weather_cols is provided.
    """
    out = df.copy()
    out = out.sort_values([perimeter_col, ts_col])

    # Target lags (strong for demand)
    lag_hours = [1, 2, 24, 48, 72, 168]
    for lag in lag_hours:
        out[f"lag_{lag}"] = out.groupby(perimeter_col)[y_col].shift(lag)

    # Rolling means on shifted series (avoid leakage)
    out["roll_mean_24"] = (
        out.groupby(perimeter_col)[y_col]
        .shift(1)
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["roll_mean_168"] = (
        out.groupby(perimeter_col)[y_col]
        .shift(1)
        .rolling(168)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Calendar
    out["hour"] = out[ts_col].dt.hour
    out["dow"] = out[ts_col].dt.dayofweek
    out["month"] = out[ts_col].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    # Perimeter ID
    out["perimeter_id"] = pd.factorize(out[perimeter_col], sort=True)[0].astype(int)

    used_weather = []
    # Weather-aware features (if available)
    if weather_cols:
        used_weather = [c for c in weather_cols if c in out.columns]

        # Convert weather cols to numeric
        for c in used_weather:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        # Degree-hour style features (very strong for electricity)
        if "temperature_2m" in used_weather:
            base = 18.0
            out["cdh_18"] = (out["temperature_2m"] - base).clip(lower=0)
            out["hdh_18"] = (base - out["temperature_2m"]).clip(lower=0)

        # Weather lags: persistence / slow dynamics
        weather_lags = [1, 2, 6, 12, 24, 48, 72, 168]
        for w in used_weather:
            for lag in weather_lags:
                out[f"{w}_lag_{lag}"] = out.groupby(perimeter_col)[w].shift(lag)

        # Rolling stats (shifted to avoid leakage)
        weather_rolls = [6, 24, 168]
        for w in used_weather:
            s = out.groupby(perimeter_col)[w].shift(1)
            for win in weather_rolls:
                out[f"{w}_roll_mean_{win}"] = (
                    s.rolling(win).mean().reset_index(level=0, drop=True)
                )
                out[f"{w}_roll_std_{win}"] = (
                    s.rolling(win).std().reset_index(level=0, drop=True)
                )

        # Degree-hour rolling features (often best)
        if "cdh_18" in out.columns and "hdh_18" in out.columns:
            for win in [24, 168]:
                s_c = out.groupby(perimeter_col)["cdh_18"].shift(1)
                s_h = out.groupby(perimeter_col)["hdh_18"].shift(1)
                out[f"cdh_18_roll_mean_{win}"] = (
                    s_c.rolling(win).mean().reset_index(level=0, drop=True)
                )
                out[f"hdh_18_roll_mean_{win}"] = (
                    s_h.rolling(win).mean().reset_index(level=0, drop=True)
                )

        # Optional interaction (helps peak shifts)
        if "temperature_2m" in used_weather:
            out["temp_x_hour"] = out["temperature_2m"] * out["hour"]

    # Force numeric dtypes for model features
    # (include all engineered weather columns automatically)
    feature_cols = [c for c in out.columns if c.startswith("lag_")] + [
        "roll_mean_24", "roll_mean_168", "hour", "dow", "month", "is_weekend", "perimeter_id"
    ]

    if used_weather:
        # raw weather cols
        feature_cols += used_weather

        # engineered weather cols: <weather>_lag_*, <weather>_roll_*
        prefixes = tuple([f"{w}_" for w in used_weather])
        feature_cols += [c for c in out.columns if c.startswith(prefixes)]

        # degree-hour extras if present
        for c in ["cdh_18", "hdh_18", "cdh_18_roll_mean_24", "cdh_18_roll_mean_168",
                  "hdh_18_roll_mean_24", "hdh_18_roll_mean_168", "temp_x_hour"]:
            if c in out.columns:
                feature_cols.append(c)

    # de-duplicate while preserving order
    feature_cols = list(dict.fromkeys(feature_cols))

    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

