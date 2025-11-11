#!/bin/bash
set -e

echo "🔁 Reiniciando aplicación..."
docker compose down
docker compose up -d
echo "✅ Reinicio completado."
