#!/bin/bash
set -e

echo "🧱 Construyendo imagen Docker para la app..."
docker compose build --no-cache
echo "✅ Build completado."
