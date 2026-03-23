import os
import sys

# Read run ID
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

# Locate mlruns folder
mlruns_path = os.getcwd()

print("Current directory:", os.listdir())

run_path = None

for folder in os.listdir(mlruns_path):
    if not os.path.isdir(folder):
        continue

    possible = os.path.join(folder, run_id)
    if os.path.exists(possible):
        run_path = possible
        break

if run_path is None:
    print("Run not found!")
    sys.exit(1)

print("Run path found:", run_path)

# Read accuracy
metric_file = os.path.join(run_path, "metrics", "accuracy")

if not os.path.exists(metric_file):
    print("Accuracy file missing!")
    sys.exit(1)

with open(metric_file, "r") as f:
    lines = f.readlines()
    accuracy = float(lines[-1].split()[1])  # ✅ FIXED

print(f"Model Accuracy: {accuracy}")

# Threshold check
threshold = 0.85

if accuracy < threshold:
    print("Model below threshold. Failing pipeline.")
    sys.exit(1)
else:
    print("Model passed threshold. Ready for deployment.")
