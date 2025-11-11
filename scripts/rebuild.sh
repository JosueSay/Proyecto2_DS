#!/bin/bash
set -e

echo "♻️ Reconstruyendo imagen desde cero..."
docker compose down --rmi all --volumes --remove-orphans
docker compose build --no-cache
echo "✅ Reconstrucción completada."
