#!/bin/bash
#
# V7 Memory Node Demo Runner
# --------------------------
# This script runs the V7 Memory Node demonstration script
# with proper environment setup

# Exit on error
set -e

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find project root (3 levels up from script directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=== V7 Memory Node Demo Runner ==="
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Check for virtual environment and activate if found
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "No virtual environment found, using system Python"
fi

# Run the memory demo script
echo "Starting memory demo..."
python src/v7/examples/memory_demo.py "$@"

# Check if the demo ran successfully
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Memory demo exited with error code: $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Memory demo completed successfully!"
exit 0 