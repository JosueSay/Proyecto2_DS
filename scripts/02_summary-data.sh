#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[1;32m"; BLUE="\033[1;34m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; RESET="\033[0m"; BOLD="\033[1m"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_DIR}/venv"
PY_BIN="${VENV_DIR}/bin/python"
[[ -x "${PY_BIN}" ]] || PY_BIN="$(command -v python3 || command -v python)"

DATA_DIR="${REPO_DIR}/data"
REPORTS_DIR="${REPO_DIR}/reports/clean"
SCRIPT_SUMMARY="${REPO_DIR}/01_data_cleaning/data_summary.py"

echo -e "\n${BLUE}${BOLD}📊 Generando resumen de datos...${RESET}\n"
echo -e "\t- data dir:\t${YELLOW}${DATA_DIR}${RESET}"
echo -e "\t- script:\t${YELLOW}${SCRIPT_SUMMARY}${RESET}\n"

PY_OUTPUT=$("${PY_BIN}" -u "${SCRIPT_SUMMARY}" 2>&1 | tee /dev/tty)
PY_EXIT_CODE=${PIPESTATUS[0]}

if [[ $PY_EXIT_CODE -ne 0 ]]; then
  echo -e "\n${RED}❌ error al generar el resumen.${RESET}"
  exit 1
fi

if [[ $PY_OUTPUT == *"CACHE_USED"* ]]; then
  echo -e "\n${YELLOW}ℹ️  resumen omitido (cache). Archivos previos en: ${YELLOW}${REPORTS_DIR}${RESET}"
elif [[ $PY_OUTPUT == *"NO_CLEAN_FILE"* ]]; then
  echo -e "\n${RED}❌ no existe data_clean.csv. ejecuta primero: scripts/00_cleaning-data.sh${RESET}"
  exit 1
elif [[ $PY_OUTPUT == *"DONE"* ]]; then
  echo -e "\n${GREEN}✅ resumen generado. ver CSVs en: ${YELLOW}${REPORTS_DIR}${RESET}"
else
  echo -e "\n${RED}❌ salida inesperada del script python.${RESET}"
  exit 1
fi
