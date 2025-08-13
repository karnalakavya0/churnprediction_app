import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and features
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("📊 Customer Churn Prediction")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.text_input("Multiple Lines")
internet_service = st.text_input("Internet Service")
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
total_charges = st.number_input("Total Charges", min_value=0.0, step=0.01)

# Prediction button
if st.button("Predict"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    df = pd.DataFrame([input_data])
    df = df.reindex(columns=feature_names, fill_value=0)
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]

    if prediction == 1:
        st.error("🚨 Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")
