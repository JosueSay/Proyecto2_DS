#!/usr/bin/env bash
set -euo pipefail

GREEN="\033[1;32m"; BLUE="\033[1;34m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; RESET="\033[0m"; BOLD="\033[1m"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_DIR}/venv"
REQ_FILE="${REPO_DIR}/requirements.txt"

echo -e "\n${BLUE}${BOLD}🚀 preparando entorno virtual...${RESET}\n"
echo -e "\t- repo:\t\t${YELLOW}${REPO_DIR}${RESET}"
echo -e "\t- requisitos:\t${YELLOW}${REQ_FILE}${RESET}\n"

if [[ -d "${VENV_DIR}" ]]; then
  echo -e "${YELLOW}venv existe, se reutiliza.${RESET}\n"
else
  echo -e "${BLUE}creando venv...${RESET}"
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel >/dev/null

if [[ -f "${REQ_FILE}" ]]; then
  echo -e "${BLUE}instalando dependencias...${RESET}\n"
  pip install -r "${REQ_FILE}"
else
  echo -e "${RED}requirements no encontrado:${RESET} ${REQ_FILE}\n"
  exit 1
fi

echo -e "\n${GREEN}${BOLD}entorno listo.${RESET}"
echo -e "\tactivar con: ${YELLOW}source ${VENV_DIR}/bin/activate${RESET}\n"
