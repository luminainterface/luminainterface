FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch first
RUN pip install torch>=2.1.0

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create volume mount points
RUN mkdir -p /data/growth_history \
    /data/concept_metrics \
    /data/base_data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data
ENV PROMETHEUS_MULTIPROC_DIR=/tmp
ENV REDIS_URL=redis://redis:6379
ENV PROMETHEUS_URL=http://prometheus:9090

# Expose the port for Prometheus metrics
EXPOSE 8000

# Run the growth monitor service
CMD ["uvicorn", "growth_monitor:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"] 