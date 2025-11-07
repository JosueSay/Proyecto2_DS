#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[1;32m"; BLUE="\033[1;34m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; RESET="\033[0m"; BOLD="\033[1m"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_DIR}/venv"
PY_BIN="${VENV_DIR}/bin/python"
[[ -x "${PY_BIN}" ]] || PY_BIN="$(command -v python3 || command -v python)"

DATA_DIR="${REPO_DIR}/data"
SCRIPT_CLEAN="${REPO_DIR}/01_data_cleaning/clean_data.py"

echo -e "\n${BLUE}${BOLD}🧹 Iniciando limpieza de datos...${RESET}\n"
echo -e "\t- data dir:\t${YELLOW}${DATA_DIR}${RESET}"
echo -e "\t- script:\t${YELLOW}${SCRIPT_CLEAN}${RESET}\n"

# ejecutar python y capturar exit + stdout
PY_OUTPUT=$("${PY_BIN}" -u "${SCRIPT_CLEAN}" 2>&1 | tee /dev/tty)
PY_EXIT_CODE=${PIPESTATUS[0]}

if [[ $PY_EXIT_CODE -ne 0 ]]; then
  echo -e "\n${RED}❌ error en limpieza de datos.${RESET}"
  exit 1
fi

if [[ $PY_OUTPUT == *"CACHE_USED"* ]]; then
  echo -e "\n${YELLOW}ℹ️  limpieza omitida (cache).${RESET}"
elif [[ $PY_OUTPUT == *"DONE"* ]]; then
  echo -e "\n${GREEN}✅ limpieza completada.${RESET}"
else
  echo -e "\n${RED}❌ salida inesperada del script python.${RESET}"
  exit 1
fi
