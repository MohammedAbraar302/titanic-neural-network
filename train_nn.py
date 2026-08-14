import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

def train_titanic_tested_nn():
    print("🚢 Loading 'tested.csv' Dataset...")
    df = pd.read_csv("data/tested.csv")
    
    # Select the required features and target based on your schema
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
    df = df[features].copy()
    
    # Handle missing values gracefully
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Convert 'Sex' text categories to binary numbers (male=0, female=1)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Drop any leftover rows with missing values
    df.dropna(inplace=True)
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    # Split dataset into training and testing partitions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Neural Networks require a standard scaler so weights and biases optimize stably
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("🧠 Initializing Neural Network (MLPClassifier)...")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(16, 8), 
        activation='relu', 
        solver='adam', 
        max_iter=100,  # Exactly 50 epochs as specified
        random_state=42,
        verbose=True
    )
    
    print("🚀 Training Neural Network over 50 epochs...")
    nn_model.fit(X_train_scaled, y_train)
    
    # Evaluate model accuracy on unseen test data
    preds = nn_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"\n🎉 Training complete! Final Test Accuracy: {acc * 100:.2f}%")
    
    # Save the trained model bundle (Model + Scaler + Feature names)
    os.makedirs("models", exist_ok=True)
    model_bundle = {
        "model": nn_model,
        "scaler": scaler,
        "features": list(X.columns)
    }
    joblib.dump(model_bundle, "models/titanic_nn_model.pkl")
    print("💾 Saved neural network bundle to `models/titanic_nn_model.pkl`")

if __name__ == "__main__":
    train_titanic_tested_nn()