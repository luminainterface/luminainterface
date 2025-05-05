#!/usr/bin/env bash
set -e

echo "🟢  Booting Lumina back-end stack…"
docker compose pull
docker compose up -d
docker compose ps

echo -e "\n✅  Services requested. Use 'docker compose logs -f' to watch them come alive." 