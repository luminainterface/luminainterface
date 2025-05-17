#!/usr/bin/env bash
set -euo pipefail

# Generate snapshot name with timestamp
SNAP="lumina-snapshot-$(date +%Y%m%d_%H%M%S)"
WORK="/tmp/$SNAP"
mkdir -p "$WORK"

echo "▶ Creating snapshot: $SNAP"
echo "▶ Working directory: $WORK"

# Save container images
echo "▶ Saving container images..."
docker compose config --services | while read -r service; do
    echo "  - Saving image for $service"
    docker save "lumina-$service" | gzip > "$WORK/${service}.tar.gz"
done

# Archive volumes
echo "▶ Archiving volumes..."
for volume in redis_data qdrant_data adapters_data grafana_data prometheus_data; do
    echo "  - Archiving $volume"
    docker run --rm \
        -v "$volume:/from" \
        -v "$WORK:/to" \
        alpine tar czf "/to/${volume}.tar.gz" -C /from .
done

# Copy configuration files
echo "▶ Copying configuration files..."
mkdir -p "$WORK/config"
cp docker-compose.yml "$WORK/config/"
cp -r services/grafana/provisioning "$WORK/config/"
cp ops/redis/redis.conf "$WORK/config/"

# Save current stream IDs
echo "▶ Capturing stream state..."
docker compose exec redis redis-cli --no-raw stream info ingest.crawl > "$WORK/streams.txt" 2>/dev/null || true
docker compose exec redis redis-cli --no-raw stream info ingest.pdf >> "$WORK/streams.txt" 2>/dev/null || true
docker compose exec redis redis-cli --no-raw stream info output.generated >> "$WORK/streams.txt" 2>/dev/null || true

# Create final archive
echo "▶ Creating final archive..."
tar czf "${SNAP}.tar.gz" -C /tmp "$SNAP"

# Cleanup
echo "▶ Cleaning up..."
rm -rf "$WORK"

echo "✅ Snapshot created: ${SNAP}.tar.gz"
echo "  - Images: $(ls -1 "$WORK"/*.tar.gz 2>/dev/null | wc -l) services"
echo "  - Volumes: redis_data, qdrant_data, adapters_data, grafana_data, prometheus_data"
echo "  - Configs: docker-compose.yml, redis.conf, Grafana provisioning"
echo "  - Stream state: streams.txt" 