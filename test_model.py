import joblib
import numpy as np
import pandas as pd

try:
    model = joblib.load('heart_disease_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
    
    # Test prediction
    # Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,Thallium
    test_input = np.array([[70,1,4,130,322,0,2,109,0,2.4,2,3,3]])
    input_scaled = scaler.transform(test_input)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    print(f"Prediction: {prediction[0]}, Probability: {probability:.2f}")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
