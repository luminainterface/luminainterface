FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install prometheus-fastapi-instrumentator first
RUN pip install prometheus-fastapi-instrumentator==6.1.0 prometheus-api-client==0.5.4

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Use the local lumina-core directory
COPY ./lumina-core ./lumina-core
RUN pip install -e ./lumina-core

# Copy the application code
COPY . .

# Set environment variables
ENV REDIS_URL=redis://redis:6379
ENV QDRANT_URL=http://qdrant:6333
ENV GRAPH_API_URL=http://graph-api:8200
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 