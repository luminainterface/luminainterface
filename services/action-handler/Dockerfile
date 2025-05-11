FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install prometheus-fastapi-instrumentator first
RUN pip install prometheus-fastapi-instrumentator==6.1.0

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set environment variables
ENV REDIS_URL=redis://redis:6379
ENV QDRANT_URL=http://qdrant:6333
ENV PROM_URL=http://prometheus:9090
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8810

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8810/health || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8810"] 