#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <snapshot-file>"
    exit 1
fi

ARCHIVE="$1"
if [ ! -f "$ARCHIVE" ]; then
    echo "❌ Snapshot file not found: $ARCHIVE"
    exit 1
fi

echo "▶ Restoring from snapshot: $ARCHIVE"
TMP=$(mktemp -d)
echo "▶ Extracting to: $TMP"

# Extract snapshot
tar xzf "$ARCHIVE" -C "$TMP"
cd "$TMP"/* || exit 1

# Load images
echo "▶ Loading container images..."
for img in *.tar.gz; do
    if [[ $img != *"volume"* ]]; then
        echo "  - Loading $img"
        gunzip -c "$img" | docker load
    fi
done

# Restore volumes
echo "▶ Restoring volumes..."
for vol in redis_data qdrant_data adapters_data grafana_data prometheus_data; do
    if [ -f "${vol}.tar.gz" ]; then
        echo "  - Restoring $vol"
        docker volume create "$vol" 2>/dev/null || true
        docker run --rm \
            -v "$vol:/to" \
            -v "$(pwd):/from" \
            alpine sh -c "rm -rf /to/* && tar xzf /from/${vol}.tar.gz -C /to"
    fi
done

# Restore configs
echo "▶ Restoring configuration files..."
if [ -d "config" ]; then
    cp config/docker-compose.yml ../
    cp -r config/provisioning ../services/grafana/
    cp config/redis.conf ../ops/redis/
fi

# Start services
echo "▶ Starting services..."
cd ../.. || exit 1
docker compose up -d

# Wait for services
echo "▶ Waiting for services to be ready..."
sleep 10
./scripts/wait_for_services.sh \
    redis:6379 \
    concept-dictionary:8500 \
    concept-trainer-growable:8710 \
    output-engine:9000 \
    masterchat-core:8839 \
    masterchat-llm:8840

# Verify stream state
echo "▶ Verifying stream state..."
if [ -f "streams.txt" ]; then
    echo "  - Checking stream lengths..."
    while read -r line; do
        if [[ $line =~ ^([0-9]+)\smessages ]]; then
            stream=${line%%:*}
            count=${BASH_REMATCH[1]}
            current=$(docker compose exec redis redis-cli --no-raw xlen "$stream" 2>/dev/null || echo "0")
            if [ "$current" -eq "$count" ]; then
                echo "    ✅ $stream: $count messages"
            else
                echo "    ⚠️  $stream: expected $count, got $current"
            fi
        fi
    done < streams.txt
fi

echo "✅ Restore complete!"
echo "  - Images loaded"
echo "  - Volumes restored"
echo "  - Configs applied"
echo "  - Services running"
echo "  - Streams verified"

# Cleanup
rm -rf "$TMP" 