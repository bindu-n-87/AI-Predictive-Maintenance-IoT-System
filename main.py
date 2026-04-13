import os
import subprocess

print("\n==============================")
print(" AI PREDICTIVE MAINTENANCE SYSTEM ")
print("==============================\n")

if not os.path.exists("data/machine_data.csv"):
    print("Generating dataset...")
    subprocess.run(["python", "src/generate_dataset.py"])
else:
    print("Dataset already exists ✔")

print("\nRunning preprocessing...")
subprocess.run(["python", "src/preprocess.py"])

print("\nTraining ML model...")
subprocess.run(["python", "src/train_model.py"])

print("\nGenerating visualizations...")
subprocess.run(["python", "src/visualize.py"])

print("\nRunning prediction system...")
subprocess.run(["python", "src/predict.py"])

print("\n==============================")
print(" SYSTEM EXECUTION COMPLETED ")
print("==============================\n")
