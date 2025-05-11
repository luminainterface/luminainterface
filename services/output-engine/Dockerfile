FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for temporary files
RUN mkdir -p /tmp/output-engine && chmod 777 /tmp/output-engine

# Set environment variable for temp directory
ENV TEMP_DIR=/tmp/output-engine

# Expose the port the app runs on
EXPOSE 9000

# Command to run the application
CMD ["python", "main.py"] 