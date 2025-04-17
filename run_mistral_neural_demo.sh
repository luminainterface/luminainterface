#!/bin/bash
# Run Mistral Neural Network Demo

# Set environment variables
DATA_DIR="data"
NEURAL_MODELS_DIR="data/neural_models"
MISTRAL_MODEL="mistral-small"
LLM_WEIGHT=0.65
NN_WEIGHT=0.35
SYSTEM_LOG_FILE="logs/system.log"

# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$NEURAL_MODELS_DIR"
mkdir -p "logs"

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH"
    exit 1
fi

# Check for required packages
echo "Checking for required packages..."

if ! python3 -c "import numpy" &> /dev/null; then
    echo "NumPy is not installed. Installing..."
    pip3 install numpy
fi

if ! python3 -c "import torch" &> /dev/null; then
    echo "PyTorch is not installed. Installing..."
    pip3 install torch
fi

# Check if Mistral API key is set
if [ -z "$MISTRAL_API_KEY" ]; then
    echo "MISTRAL_API_KEY environment variable is not set."
    echo "You can either:"
    echo "1. Set it before running this script, or"
    echo "2. Run in mock mode without the API."
    echo ""
    
    read -p "Run in mock mode? (Y/N): " MOCK_MODE
    
    if [[ "$MOCK_MODE" =~ ^[Yy]$ ]]; then
        MOCK_ARG="--mock"
    else
        read -p "Enter your Mistral API key: " MISTRAL_API_KEY
        export MISTRAL_API_KEY
        MOCK_ARG=""
    fi
else
    MOCK_ARG=""
    echo "Using Mistral API key from environment variable."
fi

# Launch neural demo with appropriate parameters
echo "Starting Mistral Neural Network Demo..."
echo ""
echo "LLM Weight: $LLM_WEIGHT"
echo "NN Weight: $NN_WEIGHT"
echo "Model: $MISTRAL_MODEL"
echo "Data Directory: $DATA_DIR"
echo "Neural Models: $NEURAL_MODELS_DIR"
echo ""

# Run the application
python3 mistral_neural_demo.py --model "$MISTRAL_MODEL" --llm-weight "$LLM_WEIGHT" --nn-weight "$NN_WEIGHT" --model-dir "$NEURAL_MODELS_DIR" $MOCK_ARG

# Check result
if [ $? -ne 0 ]; then
    echo ""
    echo "Application terminated with errors."
    echo "Check log files for details: $SYSTEM_LOG_FILE"
else
    echo ""
    echo "Application closed successfully."
fi 