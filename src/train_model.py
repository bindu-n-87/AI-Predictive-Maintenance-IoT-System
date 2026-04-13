import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# 1. Load Preprocessed Data
# -----------------------------
X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

print("Data Loaded Successfully")

# -----------------------------
# 2. Create Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# -----------------------------
# 3. Train Model
# -----------------------------
model.fit(X_train, y_train)

print("\nModel Training Completed")

# -----------------------------
# 4. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 5. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Predictive Maintenance")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# -----------------------------
# 7. Feature Importance
# -----------------------------
importances = model.feature_importances_
features = ["temperature", "vibration", "pressure", "rpm", "wear_level"]

plt.figure(figsize=(7,4))
plt.bar(features, importances)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")
plt.show()

# -----------------------------
# 8. Save Model
# -----------------------------
joblib.dump(model, "models/predictive_model.pkl")

print("\nModel Saved Successfully!")