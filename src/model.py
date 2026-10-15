"""Feature engineering, chronological split, manual gradient descent and evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "energydata_complete.csv"
OUT = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
TARGET = "target_next_10min_Wh"
FEATURES = [
    "appliances_current_Wh", "lights_Wh", "T1_C", "RH_1_pct",
    "T2_C", "RH_2_pct", "T_out_C", "RH_out_pct", "Windspeed_m_s",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"
]

def prepare_data(df):
    required = ["date", "Appliances", "lights", "T1", "RH_1", "T2",
                "RH_2", "T_out", "RH_out", "Windspeed"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if data["date"].duplicated().any():
        raise ValueError("Duplicate timestamps found. Resolve duplicates before forecasting.")
    h = data["date"].dt.hour + data["date"].dt.minute / 60
    d = data["date"].dt.dayofweek
    X = pd.DataFrame({
        "appliances_current_Wh": data["Appliances"],
        "lights_Wh": data["lights"],
        "T1_C": data["T1"], "RH_1_pct": data["RH_1"],
        "T2_C": data["T2"], "RH_2_pct": data["RH_2"],
        "T_out_C": data["T_out"], "RH_out_pct": data["RH_out"],
        "Windspeed_m_s": data["Windspeed"],
        "hour_sin": np.sin(2*np.pi*h/24), "hour_cos": np.cos(2*np.pi*h/24),
        "weekday_sin": np.sin(2*np.pi*d/7), "weekday_cos": np.cos(2*np.pi*d/7),
    })
    next_time = data["date"].shift(-1)
    y_next = data["Appliances"].shift(-1)
    valid = (next_time - data["date"]) == pd.Timedelta(minutes=10)
    result = X.loc[valid].copy()
    result.insert(0, "feature_time", data.loc[valid, "date"].to_numpy())
    result["target_time"] = next_time.loc[valid].to_numpy()
    result[TARGET] = y_next.loc[valid].to_numpy()
    result = result.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return result

class ManualLinearRegressionGD:
    """Standardized-feature multivariable linear regression trained with batch GD."""
    def __init__(self, learning_rate=0.03, iterations=2500):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0)
        self.x_std[self.x_std == 0] = 1.0
        self.y_mean = y.mean()
        self.y_std = y.std() or 1.0
        Z = (X - self.x_mean) / self.x_std
        ys = (y - self.y_mean) / self.y_std
        n, p = Z.shape
        self.w = np.zeros(p)
        self.b = 0.0
        self.loss_history = []
        for _ in range(self.iterations):
            pred = Z @ self.w + self.b
            error = pred - ys
            loss = np.mean(error**2)
            self.loss_history.append(float(loss))
            dw = (2/n) * (Z.T @ error)
            db = 2 * np.mean(error)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Z = (X - self.x_mean) / self.x_std
        return (Z @ self.w + self.b) * self.y_std + self.y_mean

def metrics(y, pred):
    return {
        "MAE_Wh": float(mean_absolute_error(y, pred)),
        "RMSE_Wh": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }

def main():
    if not CSV.exists():
        raise FileNotFoundError("Dataset missing. Run: python src/download_data.py")
    OUT.mkdir(exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(CSV)
    prepared = prepare_data(raw)
    prepared.to_csv(PROCESSED / "next_interval_dataset.csv", index=False)

    # Chronological split: never shuffle time-series rows.
    split = int(len(prepared) * 0.8)
    train, test = prepared.iloc[:split], prepared.iloc[split:]
    X_train, y_train = train[FEATURES].to_numpy(), train[TARGET].to_numpy()
    X_test, y_test = test[FEATURES].to_numpy(), test[TARGET].to_numpy()

    # Baseline: predict training-set mean for every test observation.
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    manual = ManualLinearRegressionGD().fit(X_train, y_train)
    manual_pred = manual.predict(X_test)

    sklearn_model = LinearRegression().fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)

    results = {
        "dataset_rows_raw": int(len(raw)),
        "dataset_columns_raw": int(raw.shape[1]),
        "prepared_rows": int(len(prepared)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "target": TARGET,
        "features": FEATURES,
        "learning_rate": manual.learning_rate,
        "iterations": manual.iterations,
        "initial_training_mse_standardized": float(manual.loss_history[0]),
        "final_training_mse_standardized": float(manual.loss_history[-1]),
        "training_loss_reduction_percent": float((1 - manual.loss_history[-1] / manual.loss_history[0]) * 100),
        "train_end_time": str(train["feature_time"].max()),
        "test_start_time": str(test["feature_time"].min()),
        "test_end_time": str(test["target_time"].max()),
        "manual_gradient_descent": metrics(y_test, manual_pred),
        "sklearn_linear_regression": metrics(y_test, sklearn_pred),
        "mean_baseline": metrics(y_test, baseline_pred),
        "note": "Metrics are produced by running this script on the downloaded UCI dataset."
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"iteration": np.arange(1, len(manual.loss_history) + 1), "standardized_mse": manual.loss_history}).to_csv(OUT / "training_loss_history.csv", index=False)
    pd.DataFrame({
        "target_actual_Wh": y_test,
        "manual_gd_predicted_Wh": manual_pred,
        "sklearn_predicted_Wh": sklearn_pred,
        "baseline_predicted_Wh": baseline_pred,
    }).to_csv(OUT / "test_predictions.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(manual.loss_history)
    plt.xlabel("Iteration"); plt.ylabel("MSE (standardized target)")
    plt.title("Manual Gradient Descent: Training Loss")
    plt.tight_layout(); plt.savefig(OUT / "training_loss.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, manual_pred, alpha=0.35)
    low = min(float(y_test.min()), float(manual_pred.min()))
    high = max(float(y_test.max()), float(manual_pred.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual appliance energy (Wh)")
    plt.ylabel("Predicted appliance energy (Wh)")
    plt.title("Actual vs Predicted (Manual Gradient Descent)")
    plt.tight_layout(); plt.savefig(OUT / "actual_vs_predicted.png", dpi=160); plt.close()

    print(json.dumps(results, indent=2))
    print(f"\nResults and plots saved in {OUT}")

if __name__ == "__main__":
    main()
