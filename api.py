from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Titanic Neural Network API", version="1.0")

# Load the trained model bundle on startup
bundle = joblib.load("models/titanic_nn_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

# Define incoming request body data structure using Pydantic
class Passenger(BaseModel):
    Pclass: int
    Sex: int  # 0 for male, 1 for female
    Age: float
    SibSp: int
    Parch: int
    Fare: float

@app.get("/")
def home():
    return {"message": "Titanic Neural Network API is running live!"}

@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Pack request parameters into a DataFrame matching training features
    input_data = pd.DataFrame([{
        "Pclass": passenger.Pclass,
        "Sex": passenger.Sex,
        "Age": passenger.Age,
        "SibSp": passenger.SibSp,
        "Parch": passenger.Parch,
        "Fare": passenger.Fare
    }])
    
    # Scale inputs using the exact scaler fitted during training
    scaled_input = scaler.transform(input_data)
    
    # Run prediction through the neural network
    prediction = int(model.predict(scaled_input)[0])
    probability = float(np.max(model.predict_proba(scaled_input)))
    
    result = "Survived" if prediction == 1 else "Did Not Survive"
    
    return {
        "prediction_code": prediction,
        "result": result,
        "confidence": round(probability * 100, 2)
    }