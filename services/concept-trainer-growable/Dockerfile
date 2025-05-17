# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service files
COPY main.py .
COPY model.py .
COPY routes.py .
COPY __init__.py .

# Create necessary directories
RUN mkdir -p /data/batches /adapters /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    MODEL_INPUT_SIZE=768 \
    BATCH_SIZE=32 \
    TRAINING_INTERVAL=60 \
    GROWTH_THRESHOLD=0.1 \
    API_PORT=8710 \
    METRICS_PORT=8711

# Expose ports
EXPOSE ${API_PORT} ${METRICS_PORT}

# Set volume mounts
VOLUME ["/data/batches", "/adapters", "/app/data"]

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8710"] 