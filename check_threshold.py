import os
import mlflow

mlruns_path = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file://{mlruns_path}")

# IMPORTANT: set experiment name (same as train.py)
mlflow.set_experiment("Assignment5_Classifier")

client = mlflow.tracking.MlflowClient()

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", 0)
print(f"Model Accuracy: {accuracy}")

threshold = 0.85

if accuracy < threshold:
    print("Model below threshold. Failing pipeline.")
    exit(1)
else:
    print("Model passed threshold. Ready for deployment.")
