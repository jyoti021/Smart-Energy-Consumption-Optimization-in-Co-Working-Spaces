import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Page Config
st.set_page_config(page_title="Smart Energy Analyzer", layout="wide")
st.title("Smart Energy Consumption Analyzer")
st.markdown("### Upload **ANY** energy CSV → Get instant analysis & ML predictions!")

# File uploader
uploaded_file = st.file_uploader("Upload your energy dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    except:
        st.error("Could not read file. Make sure it's a valid CSV.")
        st.stop()
else:
    st.info("No file uploaded → Showing demo with real building energy pattern")
    # Generate realistic demo data
    dates = pd.date_range("2024-01-01", periods=2000, freq='15min')
    np.random.seed(42)
    df = pd.DataFrame({
        "datetime": dates,
        "building_power_kW": np.abs(np.random.normal(8000, 4000, 2000) + np.sin(np.arange(2000)/96*np.pi)*3000),
        "cooling_load_kW": np.abs(np.random.normal(5000, 3000, 2000)),
        "temperature": 20 + 15 * np.sin(np.arange(2000)/96*np.pi) + np.random.normal(0, 3, 2000)
    })

# Auto-clean column names
df.columns = [col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]

# Auto-detect timestamp column
time_cols = ['timestamp', 'datetime', 'date', 'time', 'datetime_utc']
time_col = None
for col in df.columns:
    if any(t in col for t in time_cols):
        time_col = col
        break
if time_col is None and df.shape[1] > 0:
    time_col = df.columns[0]  # assume first column is time

df = df.rename(columns={time_col: 'timestamp'})
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp']).reset_index(drop=True)

if df.empty:
    st.error("No valid timestamp found. Check your date/time column.")
    st.stop()

# Auto-detect power/energy column (target)
power_keywords = ['power', 'kw', 'kwh', 'elec', 'load', 'consumption', 'total', 'energy', 'cool']
target_col = None
for col in df.columns:
    if col != 'timestamp' and any(k in col for k in power_keywords):
        target_col = col
        break
if target_col is None:
    target_col = df.columns[-1]  # last column as fallback

st.write(f"Detected Target Column → **{target_col.upper()}**")

# Feature Engineering (works even with ZERO extra columns!)
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))

# Lag features (very powerful!)
df = df.sort_values('timestamp').reset_index(drop=True)
df['lag_1'] = df[target_col].shift(1)
df['lag_4'] = df[target_col].shift(4)   # 1 hour ago (15-min data)
df['rolling_mean_4'] = df[target_col].rolling(4).mean()
df = df.dropna().reset_index(drop=True)

# Final features
feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos', 
                'lag_1', 'lag_4', 'rolling_mean_4']
feature_cols = [f for f in feature_cols if f in df.columns]

X = df[feature_cols]
y = df[target_col]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

st.success(f"Model Trained! MAE: {mae:,.1f} | R² Score: {r2:.3f} (closer to 1 = better)")

# Charts
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(11, 5))
    daily = df.set_index('timestamp')[target_col].resample('D').mean()
    daily.plot(ax=ax, color='#2E86AB', linewidth=2, title=f"Daily Average {target_col.upper()}")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    hourly = df.groupby('hour')[target_col].mean()
    hourly.plot(kind='bar', ax=ax2, color='#F39C6B', alpha=0.8, title="Average Hourly Pattern")
    ax2.set_ylabel("Power (kW)")
    ax2.set_xlabel("Hour of Day")
    st.pyplot(fig2)

# Prediction vs Actual
st.markdown("### Prediction vs Actual (Test Data)")
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.scatter(y_test, preds, alpha=0.6, color='#A23B72')
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax3.set_xlabel("Actual")
ax3.set_ylabel("Predicted")
ax3.set_title("Model Accuracy - How close are predictions?")
st.pyplot(fig3)

st.markdown("---")
st.success("Works with ANY energy CSV — just upload and go!")
st.balloons()

st.caption("Built with ❤️ by **Jyoti** | Smart Energy Analytics Project 2025")
st.markdown("[GitHub](https://github.com/jyoti021/Smart-Energy-Consumption-Optimization-in-Co-Working-Spaces) • Add your LinkedIn here!")
