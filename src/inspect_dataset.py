"""Basic data quality report for the UCI dataset."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "energydata_complete.csv"

def main():
    if not CSV.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/download_data.py")
    df = pd.read_csv(CSV, parse_dates=["date"])
    print("Shape:", df.shape)
    print("\nColumns and dtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nExact duplicate rows:", int(df.duplicated().sum()))
    print("Duplicate timestamps:", int(df["date"].duplicated().sum()))
    print("\nTarget summary (Appliances, Wh):\n", df["Appliances"].describe())
    delta = df["date"].sort_values().diff().dropna()
    print("\nMost common timestamp gaps:\n", delta.value_counts().head())
    print("\nFirst five rows:\n", df.head().to_string(index=False))

if __name__ == "__main__":
    main()
