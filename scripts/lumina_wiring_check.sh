#!/bin/bash
# Lumina System Wiring & Integration Automated Check
# Usage: bash scripts/lumina_wiring_check.sh

set -e

# Core service ports (edit as needed)
PORTS=(8140 8000 8401 8311 8301 8906 8501 8201)

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}✅ $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; }

summary=()

# 1. Service Health & Metrics
for port in "${PORTS[@]}"; do
  if curl -sf http://localhost:$port/health > /dev/null; then
    pass "/health on :$port"
  else
    fail "/health on :$port"; summary+=("/health fail on :$port")
  fi
  if curl -sf http://localhost:$port/metrics > /dev/null; then
    pass "/metrics on :$port"
  else
    fail "/metrics on :$port"; summary+=("/metrics fail on :$port")
  fi
done

# 2. Prometheus Targets
if curl -sf http://localhost:9090/targets | grep -q 'UP'; then
  pass "Prometheus targets show services as UP"
else
  fail "Prometheus targets not all UP"; summary+=("Prometheus targets not all UP")
fi

# 3. Grafana Dashboards
if curl -sf http://localhost:3000/api/health | grep -q 'database'; then
  pass "Grafana is running"
else
  fail "Grafana not reachable"; summary+=("Grafana not reachable")
fi

# 4. Loki Logs
if curl -sf "http://localhost:3100/ready" | grep -q 'ready'; then
  pass "Loki is running"
else
  fail "Loki not reachable"; summary+=("Loki not reachable")
fi

# 5. Jaeger Tracing
if curl -sf http://localhost:16686 | grep -q '<title>Jaeger UI'; then
  pass "Jaeger UI is running"
else
  fail "Jaeger not reachable"; summary+=("Jaeger not reachable")
fi

# 6. Redis Pub/Sub Channel
if command -v redis-cli > /dev/null; then
  if redis-cli --raw PUBSUB CHANNELS | grep -q 'crawl_request'; then
    pass "Redis pub/sub channel 'crawl_request' is active"
  else
    fail "Redis pub/sub channel 'crawl_request' not found"; summary+=("Redis pub/sub channel missing")
  fi
else
  fail "redis-cli not installed, skipping Redis check"; summary+=("redis-cli missing")
fi

# 7. Print Summary
if [ ${#summary[@]} -eq 0 ]; then
  echo -e "\n${GREEN}All checks passed!${NC}"
else
  echo -e "\n${RED}Some checks failed:${NC}"
  for item in "${summary[@]}"; do
    echo "- $item"
  done
fi 
 