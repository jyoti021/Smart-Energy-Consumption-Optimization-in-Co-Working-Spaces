import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from datetime import datetime

st.set_page_config(page_title="Energy Anomaly Detector", layout="wide")
st.title("Co-working Space Energy Anomaly Dashboard")
st.caption("Upload your data • Instant anomaly detection • 100% free")

option = st.sidebar.radio("Data source", ["Synthetic Dataset (30 days)", "Upload Your File"])

if option == "Upload Your File":
    uploaded = st.sidebar.file_uploader("Drop CSV/Excel", type=["csv", "xlsx"])
    if not uploaded:
        st.stop()
    df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
else:
    st.info("Generating 30-day hourly synthetic dataset...")
    np.random.seed(42)
    n = 24 * 30
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq='h'),
        "floor": np.random.randint(1, 4, n),
        "room": np.random.randint(1, 15, n),
        "occupancy": np.random.randint(0, 20, n),
        "temperature_outside": np.random.uniform(18, 42, n),
        "energy_kwh": np.random.uniform(0.5, 8.0, n),
    })

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

st.success(f"Loaded {df.shape[0]:,} rows")
st.dataframe(df.head(100), use_container_width=True)

# Anomaly Detection
st.subheader("Anomaly Detection")
X = df[["occupancy", "temperature_outside", "hour"]]
model = IsolationForest(contamination=0.06, random_state=42)
df["anomaly"] = model.fit_predict(X)
df["label"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

anomalies = (df["label"] == "Anomaly").sum()
st.metric("Detected Anomalies", anomalies, f"{anomalies/len(df)*100:.1f}%")

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(10,5))
    sns.scatterplot(data=df, x="occupancy", y="energy_kwh", hue="label",
                    palette=["lightgreen", "crimson"], alpha=0.8)
    plt.title("Anomalies Highlighted")
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots(figsize=(10,5))
    hourly = df.groupby("hour")["energy_kwh"].mean()
    plt.plot(hourly.index, hourly.values, marker='o', color="#6366f1", linewidth=3)
    plt.title("Average Energy per Hour"); plt.xlabel("Hour"); plt.ylabel("kWh")
    st.pyplot(fig)

csv = df.to_csv(index=False).encode()
st.download_button("Download Results with Anomalies", csv, "energy_with_anomalies.csv", "text/csv")
