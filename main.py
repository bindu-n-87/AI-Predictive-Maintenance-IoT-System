import os
import subprocess

print("\n==============================")
print(" AI PREDICTIVE MAINTENANCE SYSTEM ")
print("==============================\n")

# -----------------------------
# STEP 1: Generate Dataset (if not exists)
# -----------------------------
if not os.path.exists("data/machine_data.csv"):
    print("Generating dataset...")
    subprocess.run(["python", "src/generate_dataset.py"])
else:
    print("Dataset already exists ✔")

# -----------------------------
# STEP 2: Preprocessing
# -----------------------------
print("\nRunning preprocessing...")
subprocess.run(["python", "src/preprocess.py"])

# -----------------------------
# STEP 3: Train Model
# -----------------------------
print("\nTraining ML model...")
subprocess.run(["python", "src/train_model.py"])

# -----------------------------
# STEP 4: Visualization
# -----------------------------
print("\nGenerating visualizations...")
subprocess.run(["python", "src/visualize.py"])

# -----------------------------
# STEP 5: Prediction Demo
# -----------------------------
print("\nRunning prediction system...")
subprocess.run(["python", "src/predict.py"])

print("\n==============================")
print(" SYSTEM EXECUTION COMPLETED ")
print("==============================\n")