import streamlit as st
import requests

st.set_page_config(page_title="Titanic Deep Learning Studio", layout="centered")

st.title("🚢 Titanic Neural Network Deployed App")
st.markdown("This interface sends custom inputs to your local **FastAPI backend**, which runs inference using a **Neural Network** trained over 50 epochs.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sex_label = st.selectbox("Sex", ["Male", "Female"])
    sex = 0 if sex_label == "Male" else 1
    age = st.slider("Age", 0.0, 80.0, 28.0)

with col2:
    sibsp = st.number_input("Siblings / Spouses Aboard (SibSp)", 0, 8, 0)
    parch = st.number_input("Parents / Children Aboard (Parch)", 0, 6, 0)
    fare = st.number_input("Ticket Fare ($)", 0.0, 500.0, 32.2)

if st.button("🔮 Send Request to API", type="primary", use_container_width=True):
    payload = {
        "Pclass": int(pclass),
        "Sex": int(sex),
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": float(fare)
    }
    
    try:
        # Call the local FastAPI server
        response = requests.post("https://titanic-neural-network.onrender.com/predict", json=payload)
        
        if response.status_code == 200:
            res_json = response.json()
            outcome = res_json["result"]
            conf = res_json["confidence"]
            
            st.markdown("### API Response Result:")
            if res_json["prediction_code"] == 1:
                st.success(f"🟢 **{outcome}** (Confidence: {conf}%)")
            else:
                st.error(f"🔴 **{outcome}** (Confidence: {conf}%)")
        else:
            st.error("The API returned an unexpected response code.")
    except Exception as e:
        st.error(f"⚠️ Connection error. Make sure FastAPI is running. Details: {e}")