import mlflow
import sys

# =========================
# MLflow setup
# =========================
mlflow.set_tracking_uri("file:./mlruns")

# =========================
# Read Run ID
# =========================
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

# =========================
# Get run from MLflow
# =========================
client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

# =========================
# Get accuracy
# =========================
accuracy = run.data.metrics.get("accuracy", 0)

print(f"Model Accuracy: {accuracy}")

# =========================
# Check threshold
# =========================
threshold = 0.85

if accuracy < threshold:
    print("Model below threshold. Failing pipeline.")
    sys.exit(1)   # THIS FAILS THE PIPELINE
else:
    print("Model passed threshold. Ready for deployment.")