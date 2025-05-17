#!/bin/bash

set -e

# Create necessary directories
mkdir -p /workspace/test-results/reports
mkdir -p /workspace/test-results/logs

# Function to archive test results
archive_results() {
    local exit_code=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Create archive directory
    mkdir -p /workspace/test-results/archives
    
    # Archive test results and logs
    tar -czf "/workspace/test-results/archives/test_results_${timestamp}.tar.gz" \
        -C /workspace/test-results \
        reports/ logs/
    
    return $exit_code
}

# Wait for all services to be ready
echo "🔍 Checking service dependencies..."
/workspace/scripts/wait-for-it.sh --timeout=60 || {
    echo "❌ Service dependencies not ready"
    archive_results 1
    exit 1
}

# Run the tests
echo "🧪 Running tests..."
pytest -v \
    --junitxml=/workspace/test-results/reports/junit.xml \
    --html=/workspace/test-results/reports/report.html \
    masterchat/tests/

# Capture test exit code
TEST_EXIT_CODE=$?

# Archive results
echo "📦 Archiving test results..."
archive_results $TEST_EXIT_CODE

# Exit with test status
exit $TEST_EXIT_CODE 