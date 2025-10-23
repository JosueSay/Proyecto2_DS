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
SCRIPT_SUMMARY="${REPO_DIR}/00_data_cleaning/data_summary.py"
LOG_FILE="${REPO_DIR}/logs/data_summary.log"
CACHE_FILE="${REPO_DIR}/cache/summary_done.txt"

echo -e "\n${BLUE}${BOLD}📊 Generando resumen de datos...${RESET}\n"
echo -e "\t- Data directory:\t${YELLOW}${DATA_DIR}${RESET}"
echo -e "\t- Script:\t${YELLOW}${SCRIPT_SUMMARY}${RESET}\n"

# Ejecutar script
python "${SCRIPT_SUMMARY}"
PY_EXIT_CODE=$?

if [[ $PY_EXIT_CODE -eq 0 ]]; then
    if [[ -f "${CACHE_FILE}" ]]; then
        echo -e "\n${YELLOW}ℹ️  Resumen ya existía. Se usó log previo: ${YELLOW}${LOG_FILE}${RESET}"
    elif [[ -f "${LOG_FILE}" ]]; then
        echo -e "\n${GREEN}✅ Resumen generado con éxito. Log disponible en: ${YELLOW}${LOG_FILE}${RESET}"
    else
        echo -e "\n${RED}❌ Error: no se generó el log de resumen.${RESET}"
        exit 1
    fi
else
    echo -e "\n${RED}❌ Error al generar el resumen de datos.${RESET}\n"
    exit 1
fi
