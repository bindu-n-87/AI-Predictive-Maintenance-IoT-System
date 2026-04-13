import numpy as np
import joblib
import pandas as pd

# -----------------------------
# 1. Load Trained Model
# -----------------------------
model = joblib.load("models/predictive_model.pkl")

print("Model Loaded Successfully")

# -----------------------------
# 2. Simulated New Sensor Data (REAL-TIME MOCK)
# -----------------------------
new_data = pd.DataFrame([{
    "temperature": 85,   # high risk
    "vibration": 9,      # high vibration
    "pressure": 32,
    "rpm": 1400,
    "wear_level": 0.9
}])

# -----------------------------
# 3. Prediction
# -----------------------------
prediction = model.predict(new_data)

probability = model.predict_proba(new_data)

# -----------------------------
# 4. Output System (Industrial Style)
# -----------------------------
print("\n--- MACHINE HEALTH REPORT ---")

if prediction[0] == 1:
    print("MACHINE FAILURE RISK DETECTED!")
    print("Action: Immediate Maintenance Required")
else:
    print("MACHINE IS HEALTHY")

print("\nFailure Probability:", probability[0][1])
print("------------------------------")