import streamlit as st
import pandas as pd
import joblib

# -------- Load Model --------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Anomaly Detection using Isolation Forest")

st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0)
hour = st.slider("Transaction Hour (0–23)", 0, 23, 12)
transactions_per_day = st.number_input("Transactions Per Day", min_value=0.0)
location_change = st.selectbox("Location Changed?", [0, 1])

if st.button("Check Transaction"):
    input_data = pd.DataFrame([[
        amount,
        hour,
        transactions_per_day,
        location_change
    ]], columns=[
        "amount",
        "hour",
        "transactions_per_day",
        "location_change"
    ])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    if prediction[0] == -1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Normal")
