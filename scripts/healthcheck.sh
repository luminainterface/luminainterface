#!/bin/bash

set -e

# Default timeout in seconds
TIMEOUT=5
VERBOSE=false
JSON_OUTPUT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timeout=*)
            TIMEOUT="${1#*=}"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Function to log messages
log() {
    if [ "$VERBOSE" = true ]; then
        echo "$1"
    fi
}

# Function to check service health
check_health() {
    local url=$1
    local start_time=$(date +%s.%N)
    
    if [ "$JSON_OUTPUT" = true ]; then
        echo "{\"checking\": \"$url\", \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"}"
    else
        log "🔍 Checking $url..."
    fi
    
    # Try to get health endpoint
    if curl --silent --fail --max-time $TIMEOUT "$url" > /dev/null; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        
        if [ "$JSON_OUTPUT" = true ]; then
            echo "{\"status\": \"healthy\", \"url\": \"$url\", \"duration\": $duration}"
        else
            echo "✅ $url is healthy (${duration}s)"
        fi
        return 0
    else
        if [ "$JSON_OUTPUT" = true ]; then
            echo "{\"status\": \"unhealthy\", \"url\": \"$url\", \"error\": \"timeout or non-200 response\"}"
        else
            echo "❌ $url is not reachable"
        fi
        return 1
    fi
}

# Function to check metrics endpoint
check_metrics() {
    local url=$1
    local metrics_url="${url/\/health/\/metrics}"
    
    if [ "$JSON_OUTPUT" = true ]; then
        echo "{\"checking_metrics\": \"$metrics_url\", \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"}"
    else
        log "📊 Checking metrics at $metrics_url..."
    fi
    
    if curl --silent --fail --max-time $TIMEOUT "$metrics_url" > /dev/null; then
        if [ "$JSON_OUTPUT" = true ]; then
            echo "{\"metrics_status\": \"available\", \"url\": \"$metrics_url\"}"
        else
            echo "✅ Metrics endpoint is available"
        fi
        return 0
    else
        if [ "$JSON_OUTPUT" = true ]; then
            echo "{\"metrics_status\": \"unavailable\", \"url\": \"$metrics_url\"}"
        else
            echo "⚠️ Metrics endpoint is not available"
        fi
        return 1
    fi
}

# Build list of endpoints from environment variables
ENDPOINTS=()

# Check for service URLs in environment
if [ -n "$GRAPH_API_URL" ]; then
    ENDPOINTS+=("${GRAPH_API_URL%/}/health")
fi

if [ -n "$EVENT_MUX_URL" ]; then
    ENDPOINTS+=("${EVENT_MUX_URL%/}/health")
fi

if [ -n "$MASTERCHAT_URL" ]; then
    ENDPOINTS+=("${MASTERCHAT_URL%/}/health")
fi

# Add default endpoints if no environment variables found
if [ ${#ENDPOINTS[@]} -eq 0 ]; then
    ENDPOINTS=(
        "http://localhost:8200/health"
        "http://localhost:8301/health"
        "http://localhost:8401/health"
    )
fi

# Start JSON array if JSON output is enabled
if [ "$JSON_OUTPUT" = true ]; then
    echo "["
fi

# Check each endpoint
FAILED=0
for url in "${ENDPOINTS[@]}"; do
    if ! check_health "$url"; then
        FAILED=1
    fi
    
    # Optionally check metrics
    if [ "$VERBOSE" = true ]; then
        check_metrics "$url"
    fi
    
    # Add comma for JSON array if not the last item
    if [ "$JSON_OUTPUT" = true ] && [ "$url" != "${ENDPOINTS[-1]}" ]; then
        echo ","
    fi
done

# Close JSON array if JSON output is enabled
if [ "$JSON_OUTPUT" = true ]; then
    echo "]"
fi

# Exit with appropriate status
if [ $FAILED -eq 0 ]; then
    if [ "$JSON_OUTPUT" = false ]; then
        echo "✨ All services are healthy"
    fi
    exit 0
else
    if [ "$JSON_OUTPUT" = false ]; then
        echo "❌ Some services are unhealthy"
    fi
    exit 1
fi 