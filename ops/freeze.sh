#!/bin/bash

# Enable strict error handling
set -euo pipefail

# Load environment defaults
set -a
[ -f .env ] && . .env
set +a

# Configurable parameters with defaults
: ${SNAP_WAIT:=5}  # Default 5s wait for snapshot
: ${MAX_RETRIES:=3}  # Default 3 retries for snapshot creation
: ${QDRANT_URL:="http://localhost:6333"}  # Qdrant API endpoint
: ${ARCHIVE_COMPRESSION:=6}  # Default gzip compression level (1-9)

# Use pigz if available, fall back to gzip
GZIP_BIN=$(command -v pigz || command -v gzip)

# Enable verbose logging if DEBUG is set
[ "${DEBUG:-}" ] && set -x

# Create timestamp for the backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="lumina_freeze_${TIMESTAMP}"
SNAP_JSON="${BACKUP_DIR}/snapshot.json"
mkdir -p "$BACKUP_DIR"

# Cleanup function
cleanup() {
    if [ -d "$BACKUP_DIR" ]; then
        rm -rf "$BACKUP_DIR"
    fi
}
trap cleanup EXIT

echo "Freezing Lumina system state..."

# Build UI
echo "Building UI..."
cd ../ui
if ! npm run build; then
    echo "UI build failed"
    exit 1
fi
cd ../ops

# Export Docker images
echo "Exporting Docker images..."
if ! docker save $(docker-compose config --services | xargs -I{} echo "lumina-{}") -o "$BACKUP_DIR/images.tar"; then
    echo "Docker image export failed"
    exit 1
fi

# Create Qdrant snapshot with retry logic
echo "Creating Qdrant snapshot..."
SNAPSHOT_NAME="lumina_snapshot_${TIMESTAMP}"
for i in $(seq 1 "$MAX_RETRIES"); do
    if curl -sSL -X POST "${QDRANT_URL}/collections/lumina/snapshots/create" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"${SNAPSHOT_NAME}\"}" -o "$SNAP_JSON"; then
        break
    fi
    echo "Qdrant snapshot attempt $i failed; retrying in $((i*3))s…" >&2
    sleep $((i*3))
    [ "$i" = "$MAX_RETRIES" ] && { echo "Snapshot failed after $MAX_RETRIES attempts" >&2; exit 1; }
done

# Verify snapshot creation
[ ! -s "$SNAP_JSON" ] && { echo "Snapshot JSON empty—abort" >&2; exit 1; }

# Extract snapshot name from response
SNAP_NAME=$(jq -r '.name' "$SNAP_JSON")
[ "$SNAP_NAME" = "null" ] && { echo "Invalid snapshot response" >&2; exit 1; }

# Poll for snapshot completion
echo "Waiting for snapshot to complete..."
if [ "$SNAP_WAIT" -gt 0 ]; then
    for i in $(seq 1 "$MAX_RETRIES"); do
        if curl -sSL "${QDRANT_URL}/collections/lumina/snapshots" | jq -e ".snapshots[] | select(.name == \"$SNAP_NAME\")" > /dev/null; then
            break
        fi
        echo "Snapshot not ready, attempt $i of $MAX_RETRIES..." >&2
        sleep "$SNAP_WAIT"
        [ "$i" = "$MAX_RETRIES" ] && { echo "Snapshot never appeared in list" >&2; exit 1; }
    done
fi

# Export Qdrant data (including the snapshot)
echo "Exporting Qdrant data..."
if ! docker run --rm --volumes-from lumina-vector-db -v $(pwd)/$BACKUP_DIR:/backup alpine \
    tar czf /backup/qdrant_data.tar.gz /qdrant/storage /qdrant/snapshots; then
    echo "Qdrant data export failed"
    exit 1
fi

# Verify Qdrant archive integrity
if ! tar -tzf "$BACKUP_DIR/qdrant_data.tar.gz" > /dev/null; then
    echo "Qdrant archive integrity check failed"
    exit 1
fi

# Export Ollama models
echo "Exporting Ollama models..."
if ! docker run --rm --volumes-from lumina-llm-engine -v $(pwd)/$BACKUP_DIR:/backup alpine \
    tar czf /backup/ollama_data.tar.gz /root/.ollama; then
    echo "Ollama data export failed"
    exit 1
fi

# Export Redis data
echo "Exporting Redis data..."
if ! docker run --rm --volumes-from lumina-redis -v $(pwd)/$BACKUP_DIR:/backup alpine \
    tar czf /backup/redis_data.tar.gz /data; then
    echo "Redis data export failed"
    exit 1
fi

# Export configuration
echo "Exporting configuration..."
cp docker-compose.yml "$BACKUP_DIR/"
cp ../lumina-core/.env* "$BACKUP_DIR/" 2>/dev/null || true

# Export UI build
echo "Exporting UI build..."
cp -r ../ui/dist "$BACKUP_DIR/ui_dist"

# Create final archive with configurable compression
echo "Creating final archive..."
ARCHIVE="${BACKUP_DIR}.tar.gz"
if ! tar -I "$GZIP_BIN -${ARCHIVE_COMPRESSION}" -cf "$ARCHIVE" "$BACKUP_DIR"; then
    echo "Archive creation failed"
    exit 1
fi

# Output checksum for integrity verification
echo "Generating checksum..."
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

echo "Freeze complete: ${ARCHIVE}"
echo "Checksum: $(cat "${ARCHIVE}.sha256")" 