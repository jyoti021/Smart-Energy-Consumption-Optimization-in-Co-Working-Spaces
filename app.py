import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Smart Energy Optimizer", layout="wide")
st.title("Smart Energy Consumption Optimization in Co-Working Spaces")
st.success("Live & Fully Interactive!")

# File uploader
uploaded_file = st.file_uploader("Upload your Energy CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded yet → Using sample 30-day co-working space data")
    # Generate sample data if no upload
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=720, freq='h')
    df = pd.DataFrame({
        "timestamp": dates,
        "occupancy": np.random.randint(0, 25, 720),
        "temperature": np.random.normal(26, 6, 720),
        "ac_power_kw": np.abs(np.random.normal(6, 3, 720)),
        "lighting_kw": np.abs(np.random.normal(2, 1, 720)),
        "total_power_kw": 0
    })
    df["total_power_kw"] = df["ac_power_kw"] * (df["occupancy"] > 5) + df["lighting_kw"] + np.random.normal(1, 0.5, 720)

st.write("Data Preview", df.head())

# Feature Engineering
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day'].isin([5,6]).astype(int)

# Use common columns
features = ['occupancy', 'temperature', 'hour', 'is_weekend'] 
features = [f for f in features if f in df.columns]
target = 'total_power_kw' if 'total_power_kw' in df.columns else df.columns[-1]  # auto-detect target

if len(features) == 0:
    st.error("No valid features found. CSV must have columns like occupancy, temperature, etc.")
else:
    X = df[features].fillna(0)
    y = df[target]

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    st.success(f"Model Trained! Accuracy: {mae:.2f} kW MAE (lower = better)")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        df.groupby(df['timestamp'].dt.date)['total_power_kw'].sum().plot(ax=ax, title="Daily Energy Usage")
        ax.set_ylabel("kW")
        st.pyplot(fig)

    with col2:
        hourly = df.groupby(df['timestamp'].dt.hour)['total_power_kw'].mean()
        fig2, ax2 = plt.subplots()
        hourly.plot(kind='bar', ax=ax2, color='coral', title="Average Hourly Usage")
        ax2.set_ylabel("kW")
        st.pyplot(fig2)

    st.markdown("---")
    st.info("You can now upload real IoT/sensor CSV files and get instant predictions!")
    st.balloons()

st.write("Built with ❤️ by **Jyoti** | Data Science Fresher Portfolio Project")
st.markdown("[GitHub](https://github.com/jyoti021/Smart-Energy-Consumption-Optimization-in-Co-Working-Spaces) • [LinkedIn](https://linkedin.com/in/your-profile)")  # Update your LinkedIn
