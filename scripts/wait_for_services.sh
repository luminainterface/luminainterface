#!/bin/bash

# Wait for services to be ready
# Usage: ./wait_for_services.sh service1:port1 service2:port2 ...

set -e

for service in "$@"; do
  host=${service%:*}
  port=${service#*:}
  echo "Waiting for $host:$port..."
  
  # Try curl for 30 seconds
  for i in {1..30}; do
    if curl -s "http://$host:$port/health" > /dev/null; then
      echo "✅ $host:$port is ready"
      break
    fi
    
    if [ $i -eq 30 ]; then
      echo "❌ Timeout waiting for $host:$port"
      exit 1
    fi
    
    echo "Waiting... ($i/30)"
    sleep 1
  done
done

echo "All services are ready! 🚀" 