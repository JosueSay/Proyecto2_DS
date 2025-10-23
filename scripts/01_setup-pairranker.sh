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
VENV_DIR="${REPO_DIR}/venv"
REQ_FILE="${REPO_DIR}/01_pair-ranker-deberta/requirements_pairranker.txt"

# ===== INICIO =====
echo -e "\n${BLUE}${BOLD}🚀 Preparando entorno virtual para PairRanker...${RESET}\n"
echo -e "\t- Repositorio:\t${YELLOW}${REPO_DIR}${RESET}"
echo -e "\t- Requisitos:\t${YELLOW}${REQ_FILE}${RESET}\n"

# ===== CREAR VENV =====
if [[ -d "${VENV_DIR}" ]]; then
    echo -e "${YELLOW}Entorno virtual ya existe, se reutilizará.${RESET}\n"
else
    echo -e "${BLUE}Creando entorno virtual...${RESET}"
    python -m venv "${VENV_DIR}"
fi

# activar entorno
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel > /dev/null

# ===== INSTALAR DEPENDENCIAS =====
if [[ -f "${REQ_FILE}" ]]; then
    echo -e "${BLUE}Instalando dependencias desde requirements_pairranker.txt...${RESET}\n"
    pip install -r "${REQ_FILE}"
else
    echo -e "${RED}Archivo de requisitos no encontrado:${RESET} ${REQ_FILE}\n"
    exit 1
fi

# ===== FIN =====
echo -e "\n${GREEN}${BOLD}Entorno virtual listo.${RESET}"
echo -e "\tActivar con: ${YELLOW}source ${VENV_DIR}/bin/activate${RESET}\n"
