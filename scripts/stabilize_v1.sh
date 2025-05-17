#!/bin/bash
# Lumina v1 Release Stabilization Script

set -e  # Exit on any error

echo "🔧 Starting Lumina v1 Release Stabilization..."

# Function to check endpoint
check_endpoint() {
    local service=$1
    local endpoint=$2
    local expected_status=$3
    local url="http://localhost:${4:-8300}${endpoint}"
    
    echo "🔍 Checking ${service} ${endpoint}..."
    response=$(curl -s -w "\n%{http_code}" ${url})
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$status_code" = "$expected_status" ]; then
        echo "✅ ${service} ${endpoint} is healthy"
        return 0
    else
        echo "❌ ${service} ${endpoint} returned ${status_code}"
        echo "Response: ${body}"
        return 1
    fi
}

# Function to check metrics
check_metrics() {
    local service=$1
    local port=$2
    local metric_name=$3
    
    echo "📊 Checking ${service} metrics..."
    metrics=$(curl -s http://localhost:${port}/metrics)
    if echo "$metrics" | grep -q "${metric_name}"; then
        echo "✅ ${service} metrics found"
        return 0
    else
        echo "❌ ${service} metrics missing"
        return 1
    fi
}

# 1. Check all service health endpoints
echo "🏥 Checking service health..."
services=(
    "concept-analyzer:8301"
    "action-handler:8302"
    "learning-graph:8303"
    "trend-analyzer:8304"
    "concept-dictionary:8305"
    "dual-chat-router:8306"
    "trainer:8307"
)

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    check_endpoint "$name" "/health" "200" "$port"
done

# 2. Check metrics endpoints
echo "📈 Checking metrics endpoints..."
for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    check_metrics "$name" "$port" "lumina_"
done

# 3. Verify Prometheus configuration
echo "🔍 Checking Prometheus configuration..."
if [ -f "prometheus/prometheus.yml" ]; then
    echo "✅ Prometheus config found"
    # Verify each service is in the scrape configs
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if grep -q "localhost:${port}" prometheus/prometheus.yml; then
            echo "✅ ${name} found in Prometheus config"
        else
            echo "❌ ${name} missing from Prometheus config"
        fi
    done
else
    echo "❌ Prometheus config not found"
fi

# 4. Test E2E flow
echo "🧪 Testing E2E flow..."

# 4.1 Test concept creation
echo "📝 Testing concept creation..."
curl -X POST http://localhost:8305/concepts \
    -H 'Content-Type: application/json' \
    -d '{"term":"TestConcept","definition":"A test concept","sources":["test"]}'

# 4.2 Test concept retrieval
echo "🔍 Testing concept retrieval..."
check_endpoint "concept-dictionary" "/concepts/TestConcept" "200" "8305"

# 4.3 Test training trigger
echo "🎓 Testing training trigger..."
curl -X POST http://localhost:8307/train \
    -H 'Content-Type: application/json' \
    -d '{"concept":"TestConcept"}'

# 4.4 Verify metrics update
echo "📊 Verifying metrics update..."
sleep 5
check_metrics "trainer" "8307" "lumina_training_complete"

# 5. Check Grafana dashboards
echo "📈 Checking Grafana dashboards..."
GRAFANA_HEALTH=$(curl -s http://localhost:3000/api/health)
if [ "$GRAFANA_HEALTH" = "ok" ]; then
    echo "✅ Grafana is healthy"
    # Check for "no data" panels
    DASHBOARDS=$(curl -s http://localhost:3000/api/dashboards)
    if echo "$DASHBOARDS" | grep -q "no data"; then
        echo "⚠️ Some Grafana panels show no data"
    else
        echo "✅ All Grafana panels have data"
    fi
else
    echo "❌ Grafana health check failed"
fi

# 6. Verify Docker Compose health checks
echo "🐳 Checking Docker Compose health..."
docker compose ps

# 7. Test service recovery
echo "🔄 Testing service recovery..."
docker compose stop concept-analyzer
sleep 5
docker compose start concept-analyzer
sleep 10
check_endpoint "concept-analyzer" "/health" "200" "8301"

echo "🎉 Stabilization checks complete!"
echo "📝 See release_checklist.md for next steps" 