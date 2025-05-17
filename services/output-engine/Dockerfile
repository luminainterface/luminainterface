FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    espeak \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY services/output-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy lumina-core (the whole package) and install it in editable mode
COPY lumina-core /app/lumina-core
RUN pip install -e /app/lumina-core

# Copy output-engine service code (from services/output-engine) into /app
COPY services/output-engine /app

# Create directory for temporary files
RUN mkdir -p /tmp/output-engine && chmod 777 /tmp/output-engine

# Set environment variable for temp directory
ENV TEMP_DIR=/tmp/output-engine

# Expose the port the app runs on
EXPOSE 9000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"] 