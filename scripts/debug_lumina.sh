#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🔍 Lumina Debug Tool${NC}"
echo "====================="

# Check Docker services
echo -e "\n${YELLOW}Checking Docker services...${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "lumina|redis|qdrant|prometheus|grafana"

# Check Prometheus targets
echo -e "\n${YELLOW}Checking Prometheus targets...${NC}"
curl -s http://localhost:9091/api/v1/targets | jq -r '.data.activeTargets[] | select(.health=="down") | "DOWN: \(.labels.job) - \(.lastError)"'

# Check Redis
echo -e "\n${YELLOW}Checking Redis...${NC}"
redis-cli -p 6381 ping || echo -e "${RED}Redis not responding${NC}"

# Check Qdrant
echo -e "\n${YELLOW}Checking Qdrant...${NC}"
curl -s http://localhost:6335/health || echo -e "${RED}Qdrant not responding${NC}"

# Check service health endpoints
echo -e "\n${YELLOW}Checking service health...${NC}"
services=(
    "graph-api:8201"
    "masterchat:8301"
    "crawler:8401"
    "event-mux:8101"
    "learning-graph:8601"
    "concept-analyzer:8501"
    "action-handler:8701"
    "trend-analyzer:8801"
)

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -s "http://localhost:$port/health" > /dev/null; then
        echo -e "${GREEN}✓${NC} $name"
    else
        echo -e "${RED}✗${NC} $name"
    fi
done

# Check Grafana datasource
echo -e "\n${YELLOW}Checking Grafana datasource...${NC}"
curl -s -u admin:lumina http://localhost:3000/api/datasources | jq -r '.[] | select(.type=="prometheus") | "Prometheus: \(.url) - \(.access)"'

# Check Alertmanager silences
echo -e "\n${YELLOW}Checking Alertmanager silences...${NC}"
curl -s http://localhost:9093/api/v2/silences | jq -r '.[] | "\(.status) - \(.comment)"'

# Check memory usage
echo -e "\n${YELLOW}Checking memory usage...${NC}"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check WebSocket connections
echo -e "\n${YELLOW}Checking WebSocket connections...${NC}"
netstat -an | grep 8101 | grep ESTABLISHED | wc -l | xargs echo "Active WebSocket connections:"

echo -e "\n${YELLOW}Debug complete!${NC}" 