import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Energy Co-Working", layout="wide")

st.title("Smart Energy Consumption Optimization in Co-Working Spaces")
st.success("LIVE & WORKING PERFECTLY!")
st.balloons()

# Generate synthetic data
@st.cache_data
def get_data():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=720, freq='h')
    df = pd.DataFrame({
        "timestamp": dates,
        "occupancy": np.random.randint(0, 25, 720),
        "temperature": np.random.normal(26, 5, 720),
        "ac_power": np.abs(np.random.normal(6, 3, 720)),
        "lighting": np.abs(np.random.normal(2, 1, 720)),
        "total_power_kw": 0
    })
    df["total_power_kw"] = df["ac_power"] * (df["occupancy"] > 5) + df["lighting"] + np.random.normal(1, 0.5, 720)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    return df

df = get_data()

st.write("30 days of synthetic co-working space energy data generated!")

# Charts
col1, col2 = st.columns(2)

with col1:
    daily = df.groupby("date")["total_power_kw"].sum()
    fig1, ax = plt.subplots()
    daily.plot(kind='line', ax=ax, color='blue')
    ax.set_title("Daily Energy Consumption")
    ax.set_ylabel("kW")
    st.pyplot(fig1)

with col2:
    hourly = df.groupby("hour")["total_power_kw"].mean()
    fig2, ax = plt.subplots()
    hourly.plot(kind='bar', ax=ax, color='orange')
    ax.set_title("Average Hourly Usage")
    ax.set_ylabel("kW")
    st.pyplot(fig2)

st.markdown("---")
st.info("Full ML Model (Random Forest + Anomaly Detection) Notebook:")
st.markdown("[Run on Google Colab](https://colab.research.google.com/github/jyoti021/Smart-Energy-Consumption-Optimization-in-Co-Working-Spaces/blob/main/Smart_Energy_Consumption_Optimization_in_Co_Working_Spaces.ipynb)")

st.write("Built with ❤️ by Jyoti | Fresher Data Science Portfolio Project")
