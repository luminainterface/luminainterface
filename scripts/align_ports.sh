#!/bin/bash
set -euo pipefail

# Configuration
UI_ENV_FILE="ui/.env"
COMPOSE_FILE="docker-compose.yml"
BACKUP_SUFFIX=".bak"

echo "🔍 Starting port alignment check..."

# Create backup of UI env file if it exists
if [ -f "$UI_ENV_FILE" ]; then
    cp "$UI_ENV_FILE" "${UI_ENV_FILE}${BACKUP_SUFFIX}"
fi

# Extract ports from docker-compose.yml
echo "📖 Reading service ports from docker-compose.yml..."
API_PORT=$(grep -A 5 'hub-api:' "$COMPOSE_FILE" | grep -oP '(?<=:)\d+(?=:8000)')
REDIS_PORT=$(grep -A 5 'redis:' "$COMPOSE_FILE" | grep -oP '(?<=:)\d+(?=:6379)')
VECTOR_PORT=$(grep -A 5 'vector-db:' "$COMPOSE_FILE" | grep -oP '(?<=:)\d+(?=:6333)')
LLM_PORT=$(grep -A 5 'llm-engine:' "$COMPOSE_FILE" | grep -oP '(?<=:)\d+(?=:11434)')
MASTERCHAT_PORT=$(grep -A 5 'masterchat:' "$COMPOSE_FILE" | grep -oP '(?<=:)\d+(?=:8000)')

# Update or create UI environment file
echo "✏️ Updating UI environment variables..."
cat > "$UI_ENV_FILE" << EOF
NEXT_PUBLIC_API_URL=http://localhost:${API_PORT:-8000}
NEXT_PUBLIC_REDIS_URL=redis://localhost:${REDIS_PORT:-6379}
NEXT_PUBLIC_VECTOR_URL=http://localhost:${VECTOR_PORT:-6333}
NEXT_PUBLIC_LLM_URL=http://localhost:${LLM_PORT:-11434}
VITE_MASTERCHAT_URL=http://localhost:${MASTERCHAT_PORT:-8300}
VITE_MASTERCHAT_LOGS=http://localhost:${MASTERCHAT_PORT:-8300}/planner/logs
EOF

echo "✅ Port alignment complete!"
echo "📝 Environment file updated at: $UI_ENV_FILE"
echo "💾 Backup created at: ${UI_ENV_FILE}${BACKUP_SUFFIX}"

# Verify the update
echo "🔍 Current port configuration:"
cat "$UI_ENV_FILE"

# Create or update .env file in ui directory
cat > ui/.env << EOL
VITE_HUB_API_URL=http://localhost:8000
VITE_GRAPH_API_URL=http://localhost:8200
VITE_MASTERCHAT_URL=http://localhost:8300
EOL

echo "✅  UI environment configured with correct service ports" 