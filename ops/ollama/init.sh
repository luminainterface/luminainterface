#!/bin/bash
set -e

# Check if mistral model exists
if ! ollama list | grep -q "mistral"; then
    echo "Pulling mistral model..."
    ollama pull mistral
    echo "Mistral model pulled successfully"
else
    echo "Mistral model already exists"
fi 