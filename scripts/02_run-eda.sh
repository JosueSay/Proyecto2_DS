#!/usr/bin/env bash
set -euo pipefail

# ===== COLORES =====
GREEN="\033[1;32m"
BLUE="\033[1;34m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"
BOLD="\033[1m"

# ===== CONFIGURACIÓN =====
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_EDA="${REPO_DIR}/02_eda/eda.py"
DATA_DIR="${REPO_DIR}/data/clean"
REPORTS_DIR="${REPO_DIR}/reports/eda"
IMAGES_DIR="${REPO_DIR}/images/eda"

# ===== INICIO =====
echo -e "\n${BLUE}${BOLD}📊 Iniciando análisis exploratorio (EDA)...${RESET}\n"
echo -e "\t- Data dir:\t${YELLOW}${DATA_DIR}${RESET}"
echo -e "\t- Script:\t${YELLOW}${SCRIPT_EDA}${RESET}"
echo -e "\t- Reports:\t${YELLOW}${REPORTS_DIR}${RESET}"
echo -e "\t- Images:\t${YELLOW}${IMAGES_DIR}${RESET}\n"

# ===== EJECUCIÓN =====
PY_OUTPUT=$(python -u "${SCRIPT_EDA}" 2>&1 | tee /dev/tty)
PY_EXIT_CODE=${PIPESTATUS[0]}

if [[ $PY_EXIT_CODE -ne 0 ]]; then
    echo -e "\n${RED}❌ Error durante el análisis exploratorio.${RESET}"
    exit 1
fi

# ===== VERIFICACIÓN =====
if [[ $PY_OUTPUT == *"CACHE_USED"* ]]; then
    echo -e "\n${YELLOW}ℹ️  Análisis omitido, se reutilizó caché.${RESET}"
elif [[ $PY_OUTPUT == *"DONE"* ]]; then
    echo -e "\n${GREEN}✅ EDA completado con éxito.${RESET}"
    echo -e "\tReportes en: ${YELLOW}${REPORTS_DIR}${RESET}"
    echo -e "\tImágenes en: ${YELLOW}${IMAGES_DIR}${RESET}\n"
else
    echo -e "\n${RED}❌ Resultado inesperado del script Python.${RESET}"
    exit 1
fi
