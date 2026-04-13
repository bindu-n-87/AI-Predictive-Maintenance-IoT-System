import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/machine_data.csv")

print("Dataset Loaded")

# -----------------------------
# 2. FAILURE DISTRIBUTION
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="failure", data=df)
plt.title("Failure vs Healthy Machines")
plt.xlabel("Failure (0 = Healthy, 1 = Failure)")
plt.ylabel("Count")
plt.savefig("outputs/failure_distribution.png")
plt.show()

# -----------------------------
# 3. SENSOR TRENDS
# -----------------------------
sensors = ["temperature", "vibration", "pressure", "rpm", "wear_level"]

for sensor in sensors:
    plt.figure(figsize=(6,4))
    plt.hist(df[sensor], bins=30)
    plt.title(f"{sensor} Distribution")
    plt.xlabel(sensor)
    plt.ylabel("Frequency")
    plt.savefig(f"outputs/{sensor}_distribution.png")
    plt.show()

# -----------------------------
# 4. CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Sensor Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

print("All visualizations saved successfully!")