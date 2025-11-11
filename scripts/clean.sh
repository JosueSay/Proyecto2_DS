#!/bin/bash
set -e

echo "🧹 Limpiando todo..."
docker compose down --rmi all --volumes --remove-orphans
docker system prune -af
echo "✅ Limpieza completa."
