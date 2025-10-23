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

CFG="${REPO_DIR}/01_pair-ranker-deberta/configs/default.yaml"
TRAIN_CSV="${REPO_DIR}/data/train.csv"
OUT_DIR="${REPO_DIR}/runs/deberta_pairranker"

# ===== INFO GPU =====
echo -e "\n${BLUE}${BOLD}🚀 Iniciando entrenamiento del modelo PairRanker...${RESET}\n"
if python - <<'PY' 2>/dev/null
import torch
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Versión CUDA: {torch.version.cuda}\n")
else:
    print("Entrenamiento en CPU (GPU no disponible).\n")
PY
then
    :
else
    echo -e "${RED}Error al verificar la GPU.${RESET}\n"
fi

# ===== MOSTRAR CONFIG =====
echo -e "${BOLD}Configuración de entrenamiento:${RESET}\n"
echo -e "\t- Config:\t${YELLOW}${CFG}${RESET}"
echo -e "\t- Datos:\t${YELLOW}${TRAIN_CSV}${RESET}"
echo -e "\t- Salida:\t${YELLOW}${OUT_DIR}${RESET}\n"

# ===== CRONÓMETRO =====
START_TIME=$(date +%s)

# ===== ENTRENAMIENTO =====
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PYTHONPATH="${REPO_DIR}/pairranker_deberta/src" \
python -m train --train "${TRAIN_CSV}" --out "${OUT_DIR}" --cfg "${CFG}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# ===== FINALIZACIÓN =====
echo -e "\n${GREEN}${BOLD}Entrenamiento completado con éxito.${RESET}"
echo -e "${BOLD}Modelo guardado en:${RESET} ${YELLOW}${OUT_DIR}${RESET}"
echo -e "${BOLD}Duración total:${RESET} ${YELLOW}${DURATION}s${RESET}\n"
