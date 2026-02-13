import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("📱 Smartphone Battery Health Dashboard")

# Load dataset
df = pd.read_csv("smartphone_battery_degradation_data.csv")

# ---------------- HEATMAP SECTION ----------------
st.subheader("📊 Correlation Heatmap")

fig_heat, ax_heat = plt.subplots(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax_heat)
st.pyplot(fig_heat)

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Phone Usage Details")

age = st.number_input("Age (months)", min_value=0)
cycles = st.number_input("Charge Cycles", min_value=0)
screen = st.number_input("Screen Time (hrs/day)", min_value=0.0)
fast = st.number_input("Fast Charge %", min_value=0)
temp = st.number_input("Average Temperature (°C)", min_value=0)
discharge = st.number_input("Full Discharge Count", min_value=0)

model = joblib.load("battery_model.pkl")

if st.button("Predict & Show Graphs"):

    input_data = pd.DataFrame([{
        'age_months': age,
        'charge_cycles': cycles,
        'screen_time_hrs_day': screen,
        'fast_charge_percent': fast,
        'avg_temp_celsius': temp,
        'full_discharge_count': discharge
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Battery Health: {prediction:.2f}%")

    # ---------------- DISTRIBUTION WITH PREDICTION ----------------
    st.subheader("📈 Prediction vs Dataset Distribution")

    fig1, ax1 = plt.subplots()

    sns.histplot(df["battery_health_percent"], kde=True, bins=10,ax=ax1)

    # Red vertical line for user prediction
    ax1.axvline(prediction, color='red', linestyle='--', linewidth=2)

    ax1.set_xlabel("Battery Health (%)")
    st.pyplot(fig1)

    # ---------------- BAR VISUALIZATION ----------------
    st.subheader("📉 Predicted Health Visualization")

    fig2, ax2 = plt.subplots()
    ax2.bar(["Predicted Health"], [prediction])
    ax2.set_ylim(0, 100)
    st.pyplot(fig2)
