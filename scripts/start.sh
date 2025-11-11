#!/bin/bash
set -e

echo "🚀 Iniciando aplicación Streamlit en contenedor..."
docker compose up -d
echo "✅ Aplicación iniciada. Disponible en: http://localhost:8501"
