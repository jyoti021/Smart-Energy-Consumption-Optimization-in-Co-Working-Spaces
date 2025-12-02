import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Smart Energy Co-Working", layout="wide")
st.title("Smart Energy Consumption Optimization in Co-Working Spaces")
st.success("Live & Working Perfectly!")

# ——— Generate synthetic data if no real data ———
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_hours = 24 * 30  # 30 days
    timestamp = pd.date_range("2025-01-01", periods=n_hours, freq='h')
    
    df = pd.DataFrame({
        "timestamp": timestamp,
        "floor": np.random.randint(1, 4, n_hours),
        "room": np.random.randint(1, 15, n_hours),
        "occupancy": np.random.randint(0, 20, n_hours),
        "temperature_outside": np.random.normal(25, 8, n_hours),
        "ac_power_kw": np.random.normal(5, 2, n_hours),
        "lighting_power_kw": np.random.normal(1.5, 0.8, n_hours),
        "pc_power_kw": np.random.normal(2, 1.2, n_hours),
    })
    
    # Realistic total energy
    df["total_power_kw"] = (
        df["ac_power_kw"] * (df["occupancy"] > 0) * 1.2 +
        df["lighting_power_kw"] * (df["occupancy"] > 0) +
        df["pc_power_kw"] * (df["occupancy"] * 0.3)
    ) + np.random.normal(1, 0.5, n_hours)
    
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.date
    
    return df

df = generate_data()

st.write("Synthetic 30-day hourly dataset generated successfully!")

# Dashboard
col1, col2 = st.columns(2)

with col1:
    fig1 = px.line(df.groupby("day")["total_power_kw"].sum().reset_index(),
                   x="day", y="total_power_kw", title="Daily Energy Consumption (kW)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(df.groupby("hour")["total_power_kw"].mean().reset_index(),
                  x="hour", y="total_power_kw", title="Average Hourly Usage")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.info("Full ML notebook (Random Forest + Anomaly Detection) available here:")
st.markdown(f"[Open Interactive Notebook on Google Colab](https://colab.research.google.com/github/jyoti021/Smart-Energy-Consumption-Optimization-in-Co-Working-Spaces/blob/main/Smart_Energy_Consumption_Optimization_in_Co_Working_Spaces.ipynb)")

st.balloons()
