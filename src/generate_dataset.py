import numpy as np
import pandas as pd

np.random.seed(42)

# Number of samples (simulate industrial data)
n_samples = 5000

# Simulated IoT sensor data
temperature = np.random.normal(60, 15, n_samples)   # avg 60°C
vibration = np.random.normal(5, 2, n_samples)       # vibration level
pressure = np.random.normal(30, 8, n_samples)       # pressure units
rpm = np.random.normal(1500, 300, n_samples)        # machine speed
wear_level = np.random.uniform(0, 1, n_samples)     # degradation

# FAILURE RULE (realistic simulation logic)
failure = (
    (temperature > 80) |
    (vibration > 8) |
    (wear_level > 0.8)
).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "rpm": rpm,
    "wear_level": wear_level,
    "failure": failure
})

# Save dataset
df.to_csv("data/machine_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())