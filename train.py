import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import zipfile
import os
import mlflow
import mlflow.keras

# =========================
# MLflow setup
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Assignment5_Classifier")

# =========================
# Unzip dataset
# =========================
zip_path = "mnist_csv.zip"

if not os.path.exists("mnist_data"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("mnist_data")
    print("Unzipped successfully!")

# =========================
# Load data
# =========================
data = pd.read_csv("mnist_data/mnist_train.csv")

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Normalize
X = X / 255.0

# Split (simple split)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# =========================
# Build Classifier Model
# =========================
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# Training with MLflow
# =========================
epochs = 5
batch_size = 128

with mlflow.start_run() as run:

    # Log params
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Get final validation accuracy
    final_acc = history.history['val_accuracy'][-1]

    # Log metric
    mlflow.log_metric("accuracy", final_acc)

    print(f"Final Validation Accuracy: {final_acc}")

    # Save model
    mlflow.keras.log_model(model, "model")

    # Save run ID to file
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

print("Training finished.")