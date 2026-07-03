#!/usr/bin/env bash
# Run on the VPS from the Full-project-cybrian-qs directory (after git pull), or let GitHub Actions call it.
# Rebuilds images (installs Python/npm deps inside Docker) and restarts the stack.
#
# One-time on server: chmod +x scripts/vps-docker-update.sh
# Updates:          git pull && ./scripts/vps-docker-update.sh
#
# Optional: REBUILD_NO_CACHE=1 ./scripts/vps-docker-update.sh  (slow, full image rebuild)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f docker-compose.yml ]]; then
  echo "error: docker-compose.yml not found in $ROOT" >&2
  exit 1
fi

if [[ "${REBUILD_NO_CACHE:-0}" == "1" ]]; then
  docker compose build --no-cache --parallel
fi

docker compose pull || true
docker compose up -d --build --remove-orphans

echo "OK: stack updated in $ROOT"
