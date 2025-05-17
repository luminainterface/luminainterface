FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    build-essential \
    git \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create startup script with dependency checks
RUN echo '#!/bin/bash\n\
\n\
# Function to check service health\n\
check_service() {\n\
    local service=$1\n\
    local command=$2\n\
    local max_retries=$3\n\
    local retry_interval=$4\n\
    \n\
    echo "Waiting for $service to be ready..."\n\
    for i in $(seq 1 $max_retries); do\n\
        if $command; then\n\
            echo "$service is ready!"\n\
            return 0\n\
        fi\n\
        if [ $i -eq $max_retries ]; then\n\
            echo "$service failed to become ready after $max_retries attempts"\n\
            return 1\n\
        fi\n\
        echo "Attempt $i/$max_retries: $service not ready, waiting ${retry_interval}s..."\n\
        sleep $retry_interval\n\
    done\n\
}\n\
\n\
# Check all required services\n\
check_service "Redis" "redis-cli -h redis ping" 30 2 || exit 1\n\
check_service "Qdrant" "curl -sf http://qdrant:6333/readyz" 30 2 || exit 1\n\
\n\
# Start the application\n\
exec python -m uvicorn main:app --host 0.0.0.0 --port 8200\n\
' > /app/start.sh && chmod +x /app/start.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
EXPOSE 8200

# Add healthcheck with increased timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8200/health || exit 1

# Use the startup script as entrypoint
ENTRYPOINT ["/app/start.sh"] 