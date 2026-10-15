"""Download the UCI Appliances Energy Prediction dataset."""
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
ZIP_PATH = RAW / "appliances_energy_prediction.zip"

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    existing_csv = RAW / "energydata_complete.csv"
    if existing_csv.exists():
        print(f"Dataset already present: {existing_csv}")
        return
    if not ZIP_PATH.exists():
        print("Downloading dataset from the UCI Machine Learning Repository...")
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW)
    csvs = list(RAW.rglob("energydata_complete.csv"))
    if not csvs:
        raise FileNotFoundError("Expected energydata_complete.csv was not found in the downloaded archive.")
    print(f"Dataset ready: {csvs[0]}")
    print("Source: UCI Appliances Energy Prediction; cite the dataset and comply with CC BY 4.0.")

if __name__ == "__main__":
    main()
