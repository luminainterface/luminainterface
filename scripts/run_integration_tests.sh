#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting integration tests...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Check if services are running
echo -e "${YELLOW}Checking if services are running...${NC}"
services=("redis" "qdrant" "prometheus" "grafana" "content-extractor" "concept-linker" "retrain-listener" "rag-coordinator")
for service in "${services[@]}"; do
    if ! docker-compose ps | grep -q "$service.*Up"; then
        echo -e "${RED}Error: $service is not running${NC}"
        echo -e "${YELLOW}Starting services...${NC}"
        docker-compose up -d
        break
    fi
done

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
for service in "${services[@]}"; do
    echo -n "Waiting for $service... "
    timeout=30
    while [ $timeout -gt 0 ]; do
        if curl -s "http://localhost:$(docker-compose port $service 8000 | cut -d: -f2)/health" > /dev/null; then
            echo -e "${GREEN}ready${NC}"
            break
        fi
        sleep 1
        timeout=$((timeout - 1))
    done
    if [ $timeout -eq 0 ]; then
        echo -e "${RED}timeout${NC}"
        exit 1
    fi
done

# Run the tests
echo -e "${YELLOW}Running integration tests...${NC}"
pytest tests/test_integration.py -v --asyncio-mode=auto

# Check test results
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    
    # Print metrics summary
    echo -e "\n${YELLOW}Metrics Summary:${NC}"
    curl -s "http://localhost:9090/api/v1/query?query=up" | jq -r '.data.result[] | "\(.metric.job): \(.value[1])"'
    
    # Print stream lengths
    echo -e "\n${YELLOW}Stream Status:${NC}"
    for stream in ingest.pdf ingest.crawl concept.new output.generated rag.request rag.response retrain.dlq; do
        length=$(redis-cli xlen $stream)
        echo "$stream: $length messages"
    done
else
    echo -e "${RED}Tests failed!${NC}"
    exit 1
fi

# Optional: Clean up test data
if [ "$1" == "--cleanup" ]; then
    echo -e "\n${YELLOW}Cleaning up test data...${NC}"
    for stream in ingest.pdf ingest.crawl concept.new output.generated rag.request rag.response retrain.dlq; do
        redis-cli del $stream
    done
    echo -e "${GREEN}Cleanup complete${NC}"
fi 