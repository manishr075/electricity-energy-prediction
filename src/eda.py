"""Exploratory data analysis for the UCI appliances energy dataset."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "energydata_complete.csv"
OUT = ROOT / "outputs" / "eda"


def main():
    if not CSV.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/download_data.py")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    intervals = df["date"].diff().dropna()
    q1, q3 = df["Appliances"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_mask = (df["Appliances"] < q1 - 1.5 * iqr) | (df["Appliances"] > q3 + 1.5 * iqr)
    corr_cols = ["Appliances", "lights", "T1", "RH_1", "T2", "RH_2", "T_out", "RH_out", "Windspeed"]
    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "date_start": str(df["date"].min()),
        "date_end": str(df["date"].max()),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": int(df["date"].duplicated().sum()),
        "interval_counts": {str(k): int(v) for k, v in intervals.value_counts().items()},
        "appliances_summary_Wh": {k: float(v) for k, v in df["Appliances"].describe().items()},
        "iqr_lower_bound_Wh": float(q1 - 1.5 * iqr),
        "iqr_upper_bound_Wh": float(q3 + 1.5 * iqr),
        "potential_iqr_outlier_count": int(outlier_mask.sum()),
        "potential_iqr_outlier_percent": float(outlier_mask.mean() * 100),
        "hourly_mean_Wh": {str(int(k)): float(v) for k, v in df.groupby(df["date"].dt.hour)["Appliances"].mean().items()},
        "weekday_mean_Wh": {str(k): float(v) for k, v in df.groupby(df["date"].dt.day_name())["Appliances"].mean().items()},
        "correlation_with_appliances": {k: float(v) for k, v in df[corr_cols].corr()["Appliances"].items()},
    }
    (OUT / "eda_summary.json").write_text(json.dumps(report, indent=2))

    plt.figure(figsize=(8, 5))
    plt.hist(df["Appliances"], bins=45, edgecolor="black")
    plt.xlabel("Appliance energy per 10-minute interval (Wh)")
    plt.ylabel("Observation count")
    plt.title("Distribution of Appliance Energy Consumption")
    plt.tight_layout(); plt.savefig(OUT / "appliances_distribution.png", dpi=180); plt.close()

    daily = df.set_index("date")["Appliances"].resample("D").mean()
    plt.figure(figsize=(10, 4.8))
    plt.plot(daily.index, daily.values, linewidth=1)
    plt.xlabel("Date"); plt.ylabel("Daily mean appliance energy (Wh)")
    plt.title("Daily Mean Appliance Energy")
    plt.xticks(rotation=25); plt.tight_layout()
    plt.savefig(OUT / "daily_consumption.png", dpi=180); plt.close()

    hourly = df.groupby(df["date"].dt.hour)["Appliances"].mean()
    plt.figure(figsize=(8, 4.8))
    plt.bar(hourly.index, hourly.values)
    plt.xlabel("Hour of day"); plt.ylabel("Mean appliance energy (Wh)")
    plt.title("Mean Appliance Energy by Hour")
    plt.xticks(range(24)); plt.tight_layout()
    plt.savefig(OUT / "hourly_consumption.png", dpi=180); plt.close()

    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Matrix of Selected Variables")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout(); fig.savefig(OUT / "correlation_heatmap.png", dpi=180); plt.close(fig)

    print(json.dumps(report, indent=2))
    print(f"EDA plots and summary saved in: {OUT}")

if __name__ == "__main__":
    main()
