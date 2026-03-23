import os
import sys

# =========================
# Read Run ID
# =========================
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

# =========================
# Locate mlruns folder (FIXED PATH)
# =========================
mlruns_path = "mlruns/mlruns"  # <-- IMPORTANT FIX

# Debug (optional but useful)
print("Current directory:", os.listdir("."))
print("Inside mlruns:", os.listdir("mlruns"))

# =========================
# Find run directory
# =========================
run_path = None

for exp in os.listdir(mlruns_path):
    exp_path = os.path.join(mlruns_path, exp)

    if not os.path.isdir(exp_path):
        continue

    possible_run = os.path.join(exp_path, run_id)

    if os.path.exists(possible_run):
        run_path = possible_run
        break

if run_path is None:
    print("Run not found in mlruns folder!")
    sys.exit(1)

print(f"Run path found: {run_path}")

# =========================
# Read accuracy manually
# =========================
metric_file = os.path.join(run_path, "metrics", "accuracy")

if not os.path.exists(metric_file):
    print("Accuracy metric not found!")
    sys.exit(1)

with open(metric_file, "r") as f:
    lines = f.readlines()
    accuracy = float(lines[-1].strip().split()[-1])

print(f"Model Accuracy: {accuracy}")

# =========================
# Threshold check
# =========================
threshold = 0.85

if accuracy < threshold:
    print("Model below threshold. Failing pipeline.")
    sys.exit(1)
else:
    print("Model passed threshold. Ready for deployment.")
