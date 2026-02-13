import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("battery_model.pkl")

st.title("🔋 Smartphone Battery Health Prediction")

st.write("Enter phone usage details:")

age = st.number_input("Age (months)", min_value=0)
cycles = st.number_input("Charge Cycles", min_value=0)
screen = st.number_input("Screen Time (hrs/day)", min_value=0.0)
fast = st.number_input("Fast Charge %", min_value=0)
temp = st.number_input("Average Temperature (°C)", min_value=0)
discharge = st.number_input("Full Discharge Count", min_value=0)

if st.button("Predict Battery Health"):

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
