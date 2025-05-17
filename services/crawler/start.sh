#!/bin/bash

# Fail fast if REDIS_URL is missing
if [ -z "$REDIS_URL" ]; then
  echo "REDIS_URL not set"
  exit 1
fi

# Extract host and port from REDIS_URL
REDIS_HOST=$(echo $REDIS_URL | sed -E "s|redis://([^:/]+):([0-9]+).*|\1|")
REDIS_PORT=$(echo $REDIS_URL | sed -E "s|redis://([^:/]+):([0-9]+).*|\2|")

if [ -z "$REDIS_HOST" ] || [ -z "$REDIS_PORT" ]; then
  echo "Failed to parse REDIS_URL: $REDIS_URL"
  exit 1
fi

# Wait for dependencies
/app/scripts/wait-for-it.sh $REDIS_HOST:$REDIS_PORT -- echo "Redis is up" || exit 1
/app/scripts/wait-for-it.sh concept-dictionary:8828 -- echo "Concept Dictionary is up" || exit 1
/app/scripts/wait-for-it.sh graph-api:8200 -- echo "Graph API is up" || exit 1

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port 8400 