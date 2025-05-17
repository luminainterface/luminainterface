#!/bin/bash
# Lumina v1 Release Verification Script

set -e  # Exit on any error

echo "🔍 Starting Lumina v1 Release Verification..."

# 1. Build Services
echo "🏗️  Rebuilding services..."
docker compose build --no-cache

# 2. Start Stack
echo "🚀 Starting services..."
docker compose up -d

# 3. Wait for Health
echo "⏳ Waiting for services to be healthy..."
until curl -s http://localhost:8300/health | grep -q "planner_alive.*true"; do
    sleep 2
done

# 4. Run E2E Test
echo "🧪 Running E2E test..."
curl -X POST http://localhost:8300/tasks \
    -H 'Content-Type: application/json' \
    -d '{"crawl":["Test Concept"],"hops":1,"max_nodes":5}'

# 5. Verify Metrics
echo "📊 Checking metrics..."
sleep 10
METRICS=$(curl -s http://localhost:8300/metrics)
if echo "$METRICS" | grep -q "concept_count"; then
    echo "✅ Metrics verified"
else
    echo "❌ Metrics check failed"
    exit 1
fi

# 6. Check Grafana
echo "📈 Verifying Grafana..."
GRAFANA_HEALTH=$(curl -s http://localhost:3000/api/health)
if [ "$GRAFANA_HEALTH" = "ok" ]; then
    echo "✅ Grafana verified"
else
    echo "❌ Grafana check failed"
    exit 1
fi

# 7. Verify Concept Dictionary
echo "📚 Checking concept dictionary..."
CONCEPT_RESPONSE=$(curl -s http://localhost:8300/concepts/Test%20Concept)
if echo "$CONCEPT_RESPONSE" | grep -q "term"; then
    echo "✅ Concept dictionary verified"
else
    echo "❌ Concept dictionary check failed"
    exit 1
fi

# 8. Check Training Status
echo "🎓 Verifying training status..."
TRAINING_STATUS=$(curl -s http://localhost:8300/training/status)
if echo "$TRAINING_STATUS" | grep -q "status"; then
    echo "✅ Training status verified"
else
    echo "❌ Training status check failed"
    exit 1
fi

echo "🎉 Release verification complete!"
echo "✅ All systems operational"
echo "📝 See release_checklist.md for next steps" 