import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/machine_data.csv")

print("Dataset Loaded")
print(df.head())

X = df.drop("failure", axis=1)
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split Done")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Done")

import numpy as np

np.save("data/X_train.npy", X_train_scaled)
np.save("data/X_test.npy", X_test_scaled)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("\nPreprocessed data saved successfully!")
