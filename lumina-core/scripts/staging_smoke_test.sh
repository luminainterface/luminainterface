#!/bin/bash

# Staging smoke test for frontend-backend integration
# Usage: ./staging_smoke_test.sh <api_key>

if [ -z "$1" ]; then
    echo "Usage: $0 <api_key>"
    exit 1
fi

API_KEY=$1
STAGING_URL="http://staging.lumina.local/v1"
FE_URL="http://localhost:3000"

echo "Starting integration smoke test..."
echo "API Key: ${API_KEY:0:4}..."
echo "Backend URL: $STAGING_URL"
echo "Frontend URL: $FE_URL"
echo

# Test 1: Health check
echo "1. Testing health check..."
response=$(curl -s -w "\n%{http_code}" \
    "$STAGING_URL/health")
status_code=$(echo "$response" | tail -n1)
health_json=$(echo "$response" | sed '$d')

echo "Status: $status_code"
echo "Response: $health_json"
echo

# Test 2: CORS headers
echo "2. Testing CORS headers..."
response=$(curl -s -I -H "Origin: $FE_URL" \
    "$STAGING_URL/health")
echo "$response" | grep -i "access-control"
echo

# Test 3: OpenAI compatibility
echo "3. Testing OpenAI compatibility..."
echo "Testing non-streaming chat..."
response=$(curl -s -w "\n%{http_code}" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -X POST "$STAGING_URL/chat/completions" \
    -d '{
        "model": "phi2",
        "messages": [{"role": "user", "content": "ping"}]
    }')
status_code=$(echo "$response" | tail -n1)
echo "Status: $status_code"
echo "Response: $(echo "$response" | sed '$d')"
echo

echo "Testing streaming chat..."
response=$(curl -s -w "\n%{http_code}" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -X POST "$STAGING_URL/chat/completions" \
    -d '{
        "model": "phi2",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": true
    }')
status_code=$(echo "$response" | tail -n1)
echo "Status: $status_code"
echo "Streaming response received"
echo

# Test 4: Rate limiting
echo "4. Testing rate limits..."
echo "Making 12 chat requests (should see 429 on last 2)..."
for i in {1..12}; do
    response=$(curl -s -w "\n%{http_code}" \
        -H "X-API-Key: $API_KEY" \
        -X POST "$STAGING_URL/chat/completions" \
        -d '{"model":"phi2","messages":[{"role":"user","content":"ping"}]}')
    
    status_code=$(echo "$response" | tail -n1)
    echo "Request $i: $status_code"
    
    if [ $i -eq 10 ]; then
        echo "Rate limit should be hit on next request..."
    fi
done

echo
echo "Waiting 61 seconds for rate limit reset..."
sleep 61

# Test 5: Metrics and monitoring
echo
echo "5. Testing metrics and monitoring..."
echo "Testing metrics summary..."
response=$(curl -s -w "\n%{http_code}" \
    -H "X-API-Key: $API_KEY" \
    "$STAGING_URL/metrics/summary")
status_code=$(echo "$response" | tail -n1)
metrics_json=$(echo "$response" | sed '$d')

echo "Status: $status_code"
echo "Response: $metrics_json"
echo

echo "Testing Prometheus metrics..."
response=$(curl -s -w "\n%{http_code}" \
    "$STAGING_URL/metrics")
status_code=$(echo "$response" | tail -n1)

echo "Status: $status_code"
echo "Checking for key metrics..."
echo "$response" | grep -E "rate_limit_hits_total|request_latency_seconds|cache_operations_total"
echo

# Test 6: Known limitations
echo "6. Testing known limitations..."
echo "Testing unsupported model..."
response=$(curl -s -w "\n%{http_code}" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -X POST "$STAGING_URL/chat/completions" \
    -d '{
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "ping"}]
    }')
status_code=$(echo "$response" | tail -n1)
echo "Status: $status_code"
echo "Response: $(echo "$response" | sed '$d')"
echo

echo "Smoke test complete. Expected results:"
echo "1. Health check: 200 OK with service statuses"
echo "2. CORS: Access-Control-Allow-Origin: $FE_URL"
echo "3. OpenAI compatibility: 200 OK for both streaming and non-streaming"
echo "4. Rate limits: First 10 = 200, last 2 = 429"
echo "5. Metrics: JSON with latency_p95_ms and service statuses"
echo "6. Limitations: 400 for unsupported model"
echo
echo "Next steps:"
echo "1. Check Grafana dashboard for rate limit metrics"
echo "2. Verify frontend can connect without CORS errors"
echo "3. Confirm service badges update based on /health"
echo "4. Test latency bars with /metrics/summary data"
echo "5. Verify OpenAI SDK compatibility in frontend" 