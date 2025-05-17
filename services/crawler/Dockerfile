# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy lumina-core first and install it
COPY services/crawler/lumina-core /app/lumina-core
WORKDIR /app/lumina-core
RUN pip install -e .

# Go back to crawler directory and copy requirements
WORKDIR /app/services/crawler
COPY services/crawler/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/crawler .

# Create directory for training data
RUN mkdir -p /app/training_data

# Set environment variables
ENV PYTHONPATH=/app
ENV REDIS_URL=redis://:02211998@redis:6379
ENV QDRANT_URL=http://qdrant:6333
ENV OLLAMA_URL=http://ollama:11434
ENV OLLAMA_MODEL=nomic-embed-text
ENV TRAINING_DATA_PATH=/app/training_data
ENV CHUNK_SIZE=1000
ENV CHUNK_OVERLAP=200
ENV BATCH_SIZE=32
ENV PROCESS_INTERVAL=3600
ENV INQUIRY_CHECK_INTERVAL=300
ENV RETRAIN_INTERVAL=100

# Expose port
EXPOSE 8400

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8400"] 