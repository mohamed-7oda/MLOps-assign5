# Base Image
FROM python:3.10-slim

# Working Directory
WORKDIR /app

# Accept RUN_ID from pipeline
ARG RUN_ID

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Simulate model download
RUN echo "Downloading model for RUN_ID=${RUN_ID}"

# Default command
CMD ["echo", "Model container is ready"]