from pathlib import Path
import pandas as pd

def load_dataset(path: str, ts_col: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p.resolve()}")

    df = pd.read_csv(p)

    # Timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    # Force numeric Consumption_MWh (handle commas / spaces)
    if "Consumption_MWh" in df.columns:
        s = df["Consumption_MWh"].astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        df["Consumption_MWh"] = pd.to_numeric(s, errors="coerce")

    # Hour should be int
    if "Hour" in df.columns:
        df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").astype("Int64")

    # Clean perimeter
    if "Perimeter" in df.columns:
        df["Perimeter"] = df["Perimeter"].astype(str).str.strip()

    df = df.dropna(subset=["Consumption_MWh", "Perimeter"])
    return df
