#!/bin/bash

# Wait for Qdrant to be ready
until curl -s http://vector-db:6333/health; do
    echo "Waiting for Qdrant..."
    sleep 1
done

# Create chat_long collection if it doesn't exist
curl -X PUT 'http://vector-db:6333/collections/chat_long' \
    -H 'Content-Type: application/json' \
    -d '{
        "vectors": {
            "size": 384,
            "distance": "Cosine"
        }
    }'

echo "Qdrant initialization complete" 