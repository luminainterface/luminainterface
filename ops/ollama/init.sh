#!/bin/bash
set -e

# Wait for Ollama to be ready
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "Waiting for Ollama to be ready..."
    sleep 5
done

# Pull Mistral model
echo "Pulling Mistral model..."
    ollama pull mistral

echo "Initialization complete!" 