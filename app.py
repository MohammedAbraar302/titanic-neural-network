import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to check prediction probabilities using the trained neural network model.")

# Load the trained model bundle
@st.cache_resource
def load_bundle():
    bundle = joblib.load("models/titanic_nn_model.pkl")
    return bundle["model"], bundle["scaler"], bundle["features"]

model, scaler, features = load_bundle()

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
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare]], columns=features)
    scaled_input = scaler.transform(input_data)
    prediction = int(model.predict(scaled_input)[0])
    probability = float(np.max(model.predict_proba(scaled_input)))
    
    result = "Survived" if prediction == 1 else "Did Not Survive"
    
    if prediction == 1:
        st.success(f"🎉 **Survived!** (Confidence: {probability * 100:.2f}%)")
    else:
        st.error(f"⚠️ **Did not survive.** (Confidence: {probability * 100:.2f}%)")