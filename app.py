import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to check prediction probabilities using the trained neural network model.")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# UI Inputs
st.subheader("Passenger Information")
pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], format_func=lambda x: f"Class {x}")
sex_input = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex_input == "Female" else 0
age = st.slider("Age", 1.0, 80.0, 28.0)
sibsp = st.selectbox("Siblings/Spouses Aboard", [0, 1, 2, 3, 4, 5, 6, 7, 8])
parch = st.selectbox("Parents/Children Aboard", [0, 1, 2, 3, 4, 5])
fare = st.slider("Fare Paid (£)", 0.0, 500.0, 32.0)

if st.button("Predict Survival", type="primary"):
    payload = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare
    }
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        prediction = result["prediction_code"]
        probability = result["confidence"] / 100
        result_text = result["result"]
        
        if prediction == 1:
            st.success(f"🎉 **Survived!** (Confidence: {probability * 100:.2f}%)")
        else:
            st.error(f"⚠️ **Did not survive.** (Confidence: {probability * 100:.2f}%)")
            
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        st.info(f"Make sure the API is running at {API_URL}")