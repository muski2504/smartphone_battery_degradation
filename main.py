import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.title("🔋 Smartphone Battery Health ML Dashboard")

# Load dataset
df = pd.read_csv("smartphone_battery_degradation_data.csv")

# Sidebar
option = st.sidebar.selectbox(
    "Choose Section",
    ["Data Analysis", "Model Performance", "Prediction"]
)

# ---------------- DATA ANALYSIS ----------------
if option == "Data Analysis":

    st.subheader("📊 Correlation Heatmap")

    fig1, ax1 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1)
    st.pyplot(fig1)

    st.subheader("📈 Battery Health Distribution")

    fig2, ax2 = plt.subplots()
    sns.histplot(df["battery_health_percent"], kde=True, ax=ax2)
    st.pyplot(fig2)

# ---------------- MODEL PERFORMANCE ----------------
elif option == "Model Performance":

    model = joblib.load("battery_model.pkl")

    X = df.drop(columns='battery_health_percent')
    y = df['battery_health_percent']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.write(f"### R² Score: {r2:.4f}")
    st.write(f"### RMSE: {rmse:.4f}")

    st.subheader("📉 Actual vs Predicted")

    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test, y_pred)
    ax3.set_xlabel("Actual Battery Health")
    ax3.set_ylabel("Predicted Battery Health")
    ax3.plot([50,100],[50,100],'r--')
    st.pyplot(fig3)

# ---------------- PREDICTION ----------------
else:

    model = joblib.load("battery_model.pkl")

    st.subheader("Enter Phone Usage Details")

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

        fig4, ax4 = plt.subplots()
        ax4.bar(["Predicted Health"], [prediction])
        ax4.set_ylim(0, 100)
        st.pyplot(fig4)
