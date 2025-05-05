#!/usr/bin/env bash
set -euo pipefail
RED='\033[0;31m'; GRN='\033[0;32m'; NC='\033[0m'

# --- configurable ports (match docker-compose) ---
GRAPH_API=${GRAPH_API:-http://localhost:8201}
MASTERCHAT=${MASTERCHAT:-http://localhost:8301}
EVENT_MUX_WS=${EVENT_MUX_WS:-ws://localhost:8101/ws}
CRAWLER_METRICS=${CRAWLER_METRICS:-http://localhost:8401/metrics}

echo -e "${GRN}▶ Smoke test: Lumina end-to-end${NC}"

# 1. enqueue a tiny crawl task
echo -n "  • enqueue crawl… "
curl -sf -X POST "$MASTERCHAT/tasks" \
  -H 'Content-Type: application/json' \
  -d '{"crawl":["Ping"],"hops":0,"max_nodes":5}' >/dev/null
echo "done"

# 2. wait for a node.add event via event-mux logs
echo -n "  • wait for node.add event "
for i in {1..12}; do
  if docker compose logs --tail=20 event-mux | grep -q "node.add"; then
    echo -e "${GRN}OK${NC}"
    break
  fi
  printf '.'; sleep 5
  [ $i -eq 12 ] && { echo -e "${RED}TIMEOUT${NC}"; exit 1; }
done

# 3. check metrics endpoints
echo -n "  • crawler /metrics reachable "
curl -sf "$CRAWLER_METRICS" | grep -q "crawler_pages_total" \
  && echo -e "${GRN}OK${NC}" || { echo -e "${RED}FAIL${NC}"; exit 1; }

echo -n "  • graph-api /metrics.summary reachable "
curl -sf "$GRAPH_API/metrics/summary" | jq '.nodes' >/dev/null \
  && echo -e "${GRN}OK${NC}" || { echo -e "${RED}FAIL${NC}"; exit 1; }

# 4. health endpoints
for svc in graph-api masterchat; do
  url=$([ "$svc" = "graph-api" ] && echo $GRAPH_API || echo $MASTERCHAT)
  echo -n "  • $svc health "
  curl -sf "$url/health" >/dev/null && echo -e "${GRN}OK${NC}" || {
    echo -e "${RED}FAIL${NC}"; exit 1; }
done

echo -e "${GRN}✅  Smoke test passed — pipeline alive${NC}" 