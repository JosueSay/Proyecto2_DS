#!/bin/bash
set -e

echo "🛑 Deteniendo contenedores..."
docker compose down
echo "✅ Contenedores detenidos."
