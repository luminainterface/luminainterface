#!/bin/bash

# Maximum number of attempts
MAX_ATTEMPTS=30
# Delay between attempts in seconds
DELAY=5

# Function to check service health
check_service() {
    local service=$1
    local url=$2
    local attempt=1
    
    echo "Checking $service health..."
    
    while [ $attempt -le $MAX_ATTEMPTS ]; do
        if curl -s -f "$url/health" > /dev/null; then
            echo "$service is healthy"
            return 0
        fi
        
        echo "Attempt $attempt: $service not ready, waiting ${DELAY}s..."
        sleep $DELAY
        attempt=$((attempt + 1))
    done
    
    echo "$service failed to become healthy after $MAX_ATTEMPTS attempts"
    return 1
}

# Check Redis
echo "Checking Redis..."
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    if redis-cli -h localhost -p 6381 ping > /dev/null 2>&1; then
        echo "Redis is healthy"
        break
    fi
    echo "Attempt $attempt: Redis not ready, waiting ${DELAY}s..."
    sleep $DELAY
    attempt=$((attempt + 1))
done

if [ $attempt -gt $MAX_ATTEMPTS ]; then
    echo "Redis failed to become healthy"
    exit 1
fi

# Check Qdrant
echo "Checking Qdrant..."
attempt=1
while [ $attempt -le $MAX_ATTEMPTS ]; do
    if curl -s -f "http://localhost:6335/readyz" > /dev/null; then
        echo "Qdrant is healthy"
        break
    fi
    echo "Attempt $attempt: Qdrant not ready, waiting ${DELAY}s..."
    sleep $DELAY
    attempt=$((attempt + 1))
done

if [ $attempt -gt $MAX_ATTEMPTS ]; then
    echo "Qdrant failed to become healthy"
    exit 1
fi

# Check services
services=(
    "graph-api:http://localhost:8201"
    "crawler:http://localhost:8401"
    "masterchat:http://localhost:8301"
)

for service in "${services[@]}"; do
    IFS=: read -r name url <<< "$service"
    if ! check_service "$name" "$url"; then
        exit 1
    fi
done

echo "All services are healthy!"
exit 0 