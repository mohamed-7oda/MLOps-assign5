import mlflow
import os
import sys

# Set tracking URI
mlruns_path = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

# Read Run ID
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

client = mlflow.tracking.MlflowClient()

# 🔥 Search ALL experiments
found = False
for exp in client.search_experiments():
    try:
        run = client.get_run(run_id)
        found = True
        break
    except:
        continue

if not found:
    print("Run not found in any experiment!")
    sys.exit(1)

# Get accuracy
accuracy = run.data.metrics.get("accuracy", 0)
print(f"Model Accuracy: {accuracy}")

# Threshold check
threshold = 0.85

if accuracy < threshold:
    print("Model below threshold. Failing pipeline.")
    sys.exit(1)
else:
    print("Model passed threshold. Ready for deployment.")
