#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[1;32m"
BLUE="\033[1;34m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"
BOLD="\033[1m"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_DIR}/venv/bin/activate"

DATA_DIR="${REPO_DIR}/data"
SCRIPT_CLEAN="${REPO_DIR}/00_data_cleaning/clean_data.py"

echo -e "\n${BLUE}${BOLD}🧹 Iniciando limpieza de datos...${RESET}\n"
echo -e "\t- Data directory:\t${YELLOW}${DATA_DIR}${RESET}"
echo -e "\t- Script:\t${YELLOW}${SCRIPT_CLEAN}${RESET}\n"

PY_OUTPUT=$(python "${SCRIPT_CLEAN}")
PY_EXIT_CODE=$?

if [[ $PY_EXIT_CODE -ne 0 ]]; then
    echo -e "\n${RED}❌ Error en limpieza de datos.${RESET}"
    exit 1
fi

if [[ $PY_OUTPUT == *"CACHE_USED"* ]]; then
    echo -e "\n${YELLOW}ℹ️  Limpieza omitida porque ya se realizó previamente.${RESET}"
elif [[ $PY_OUTPUT == *"DONE"* ]]; then
    echo -e "\n${GREEN}✅ Limpieza completada con éxito.${RESET}"
else
    echo -e "\n${RED}❌ Resultado inesperado del script Python.${RESET}"
    exit 1
fi
