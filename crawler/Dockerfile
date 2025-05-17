FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    procps \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire crawler package
COPY . /app/crawler/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create startup script
RUN echo '#!/bin/bash\n\
\n\
# Function to check if a service is healthy\n\
check_service() {\n\
    local service=$1\n\
    local host=$2\n\
    local port=$3\n\
    local max_retries=30\n\
    local retry_count=0\n\
\n\
    echo "Waiting for $service to be ready..."\n\
    while [ $retry_count -lt $max_retries ]; do\n\
        if nc -z "$host" "$port" > /dev/null 2>&1; then\n\
            echo "$service is ready!"\n\
            return 0\n\
        fi\n\
        echo "Waiting for $service... (attempt $((retry_count + 1))/$max_retries)"\n\
        sleep 2\n\
        retry_count=$((retry_count + 1))\n\
    done\n\
\n\
    echo "Error: $service failed to become ready after $max_retries attempts"\n\
    return 1\n\
}\n\
\n\
# Wait for required services\n\
check_service "Redis" "redis" "6379" || exit 1\n\
check_service "Qdrant" "qdrant" "6333" || exit 1\n\
\n\
# Start FastAPI application\n\
echo "Starting FastAPI application..."\n\
exec python -m uvicorn crawler.app.main:app --host 0.0.0.0 --port 8400\n\
' > /app/start.sh && chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8400/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/start.sh"] 