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
source "${REPO_DIR}/venv/bin/activate"

MODEL_DIR="${REPO_DIR}/runs/deberta_pairranker"
TEST_CSV="${REPO_DIR}/data/test.csv"
OUT_CSV="${REPO_DIR}/submission_pairranker.csv"

# ===== INFERENCIA =====
echo -e "\n${BLUE}${BOLD}🔍 Iniciando inferencia PairRanker...${RESET}\n"
echo -e "\t- Modelo:\t${YELLOW}${MODEL_DIR}${RESET}"
echo -e "\t- Test CSV:\t${YELLOW}${TEST_CSV}${RESET}"
echo -e "\t- Output:\t${YELLOW}${OUT_CSV}${RESET}\n"

PYTHONPATH="${REPO_DIR}/pairranker_deberta/src" \
python -m infer --model_dir "${MODEL_DIR}" --test "${TEST_CSV}" --out "${OUT_CSV}"

# ===== RESULTADO =====
if [[ -f "${OUT_CSV}" ]]; then
    echo -e "\n${GREEN}Inferencia completada con éxito.${RESET}"
    echo -e "${BOLD}Archivo guardado en:${RESET} ${YELLOW}${OUT_CSV}${RESET}\n"
else
    echo -e "\n${RED}Error: no se generó el archivo de salida.${RESET}\n"
    exit 1
fi
