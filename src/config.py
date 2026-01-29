from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class BacktestConfig:
    # Data
    data_path: str = "data/processed/features_hourly.csv"  # or a merged file later

    # Columns
    ts_col: str = "Timestamp"
    y_col: str = "Consumption_MWh"
    perimeter_col: str = "Perimeter"

    # Task definition
    lookback_hours: int = 168
    horizons: List[int] = None  # set in __post_init__ style below
    max_horizon: int = 168

    # Splits (locked)
    train_start: str = "2023-01-01"
    train_end: str = "2024-06-30 23:00:00"

    val_start: str = "2024-07-01 00:00:00"
    val_end: str = "2024-12-31 23:00:00"

    test_start: str = "2025-01-01 00:00:00"
    test_end: str = "2025-12-12 23:00:00"

    # Backtesting anchors
    stride: str = "7D"  # weekly anchors

    # Baseline settings
    seasonal_lag_hours: int = 168  # same hour last week

    weather_cols: List[str] = None

    def __post_init__(self):
        # not used because frozen=True; use helper below
        pass

def default_cfg() -> BacktestConfig:
    short = list(range(1, 25))
    medium = [24, 48, 72, 96, 120, 144, 168]
    horizons = sorted(set(short + medium))
    weather_cols = [
        "temperature_2m",
        "wind_speed_10m",
        "relative_humidity_2m",
        "shortwave_radiation",
    ]
    return BacktestConfig(horizons=horizons, max_horizon=max(horizons), weather_cols=weather_cols)
