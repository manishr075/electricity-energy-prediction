"""Streamlit dashboard for the energy prediction project."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from src.model import prepare_data, ManualLinearRegressionGD, FEATURES, TARGET

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "data" / "raw" / "energydata_complete.csv"
METRICS = ROOT / "outputs" / "metrics.json"
LOSS = ROOT / "outputs" / "training_loss.png"
ACTUAL = ROOT / "outputs" / "actual_vs_predicted.png"

st.set_page_config(page_title="EnergyWise | Consumption Predictor", page_icon="⚡", layout="wide")
st.title("EnergyWise")
st.caption("Electricity consumption prediction, calculus visualisation and cost scenarios")
st.info("Educational prototype using the UCI Appliances Energy Prediction dataset. It is not a utility-grade forecast or an official electricity bill calculator.")

if not CSV.exists():
    st.warning("Dataset not found. From the project folder, run `python src/download_data.py`, then `python src/model.py`.")
    st.stop()

if not METRICS.exists():
    st.warning("Model results are not available yet. Run `python src/model.py` from the project folder.")
    st.stop()

metrics = json.loads(METRICS.read_text())

@st.cache_data
def load_source_data(file_mtime):
    return pd.read_csv(CSV, parse_dates=["date"])

@st.cache_resource
def load_trained_model(file_mtime):
    source = pd.read_csv(CSV)
    prepared = prepare_data(source)
    split = int(len(prepared) * 0.8)
    train = prepared.iloc[:split]
    return ManualLinearRegressionGD().fit(
        train[FEATURES].to_numpy(), train[TARGET].to_numpy()
    )

file_mtime = CSV.stat().st_mtime
df = load_source_data(file_mtime)
model = load_trained_model(file_mtime)

st.subheader("Model performance")
m1, m2, m3 = st.columns(3)
m1.metric("Test MAE", f"{metrics['manual_gradient_descent']['MAE_Wh']:.2f} Wh")
m2.metric("Test RMSE", f"{metrics['manual_gradient_descent']['RMSE_Wh']:.2f} Wh")
m3.metric("Test R²", f"{metrics['manual_gradient_descent']['R2']:.3f}")
st.caption("These are held-out test metrics from the local model run, not guaranteed future performance.")

st.subheader("Predict the next 10-minute interval")
latest = df.iloc[-1]
with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    current = c1.number_input("Current appliance energy (Wh)", min_value=0.0, value=float(latest["Appliances"]))
    lights = c2.number_input("Lighting energy (Wh)", min_value=0.0, value=float(latest["lights"]))
    t1 = c3.number_input("Indoor temperature T1 (°C)", value=float(latest["T1"]))
    rh1 = c1.number_input("Indoor humidity RH1 (%)", min_value=0.0, max_value=100.0, value=float(latest["RH_1"]))
    t2 = c2.number_input("Indoor temperature T2 (°C)", value=float(latest["T2"]))
    rh2 = c3.number_input("Indoor humidity RH2 (%)", min_value=0.0, max_value=100.0, value=float(latest["RH_2"]))
    tout = c1.number_input("Outdoor temperature (°C)", value=float(latest["T_out"]))
    rhout = c2.number_input("Outdoor humidity (%)", min_value=0.0, max_value=100.0, value=float(latest["RH_out"]))
    wind = c3.number_input("Wind speed (m/s)", min_value=0.0, value=float(latest["Windspeed"]))
    hour = c1.slider("Hour of day", 0, 23, int(pd.to_datetime(latest["date"]).hour))
    weekday = c2.selectbox("Day of week", list(range(7)), index=int(pd.to_datetime(latest["date"]).dayofweek), format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])
    tariff = c3.number_input("Assumed tariff (₹/kWh)", min_value=0.0, value=8.0, step=0.5)
    submitted = st.form_submit_button("Predict consumption")

if submitted:
    row = pd.DataFrame([{
        "appliances_current_Wh": current, "lights_Wh": lights,
        "T1_C": t1, "RH_1_pct": rh1, "T2_C": t2, "RH_2_pct": rh2,
        "T_out_C": tout, "RH_out_pct": rhout, "Windspeed_m_s": wind,
        "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
        "weekday_sin": np.sin(2*np.pi*weekday/7), "weekday_cos": np.cos(2*np.pi*weekday/7),
    }])[FEATURES]
    prediction = max(0.0, float(model.predict(row.to_numpy())[0]))
    cost = prediction / 1000 * tariff
    a, b = st.columns(2)
    a.metric("Predicted next-interval energy", f"{prediction:.2f} Wh")
    b.metric("Estimated cost for that interval", f"₹{cost:.4f}")
    st.caption("The predicted value is for a 10-minute interval. Cost estimates use your assumed tariff and exclude fixed charges, taxes and slab structures.")

st.subheader("What-if savings simulator")
w1, w2 = st.columns(2)
base_wh = w1.number_input("Energy scenario baseline (Wh)", min_value=0.0, value=500.0, step=25.0)
rate = w2.number_input("Tariff for scenario (₹/kWh)", min_value=0.0, value=8.0, step=0.5, key="scenario_rate")
reduction = st.slider("Hypothetical consumption reduction", 0, 50, 20, 5)
before = base_wh / 1000 * rate
after = base_wh * (1 - reduction/100) / 1000 * rate
s1, s2, s3 = st.columns(3)
s1.metric("Cost before", f"₹{before:.4f}")
s2.metric("Cost after", f"₹{after:.4f}")
s3.metric("Estimated difference", f"₹{before-after:.4f}")
st.caption("This is arithmetic on a hypothetical reduction, not a claim that the model can cause or guarantee savings.")

st.subheader("Training and evaluation plots")
left, right = st.columns(2)
if LOSS.exists():
    left.image(str(LOSS), caption="Training loss over gradient-descent iterations", use_container_width=True)
if ACTUAL.exists():
    right.image(str(ACTUAL), caption="Held-out actual versus predicted consumption", use_container_width=True)

st.subheader("Compare models")
comparison = pd.DataFrame({
    "Model": ["Mean baseline", "Manual gradient descent", "Scikit-learn linear regression"],
    "MAE (Wh)": [metrics["mean_baseline"]["MAE_Wh"], metrics["manual_gradient_descent"]["MAE_Wh"], metrics["sklearn_linear_regression"]["MAE_Wh"]],
    "RMSE (Wh)": [metrics["mean_baseline"]["RMSE_Wh"], metrics["manual_gradient_descent"]["RMSE_Wh"], metrics["sklearn_linear_regression"]["RMSE_Wh"]],
    "R²": [metrics["mean_baseline"]["R2"], metrics["manual_gradient_descent"]["R2"], metrics["sklearn_linear_regression"]["R2"]],
})
st.dataframe(comparison, use_container_width=True, hide_index=True)
st.caption("The dataset describes one building in Belgium over a limited period. Results may not generalise to Indian homes or other buildings.")
