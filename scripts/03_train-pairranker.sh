#!/usr/bin/env bash
set -euo pipefail

# ===== colores =====
GREEN="\033[1;32m"; BLUE="\033[1;34m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; RESET="\033[0m"; BOLD="\033[1m"

# ===== rutas =====
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_DIR}/venv/bin/activate"
SRC_DIR="${REPO_DIR}/m_pair-ranker-deberta/src"
CFG="${REPO_DIR}/m_pair-ranker-deberta/configs/default.yaml"
TRAIN_CSV="${REPO_DIR}/data/clean/data_clean_aug.csv"      # o train_strat.csv si prefieres estratificado
VALID_CSV="${REPO_DIR}/data/clean/valid_strat.csv"         # se pasa solo si existe
OUT_DIR="${REPO_DIR}/results/deberta/run_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}" "${REPO_DIR}/reports/deberta"

# ===== entorno =====
# shellcheck disable=SC1090
source "${VENV}"
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HOME="${REPO_DIR}/.hf_cache"

# ===== info gpu =====
echo -e "\n${BLUE}${BOLD}🚀 Entrenamiento PairRanker (DeBERTa)${RESET}\n"
python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | BF16:", torch.cuda.is_bf16_supported())
else:
    print("CPU (GPU no disponible)")
PY

# ===== resumen cfg =====
echo -e "${BOLD}config:${RESET} ${YELLOW}${CFG}${RESET}"
echo -e "${BOLD}train:${RESET}  ${YELLOW}${TRAIN_CSV}${RESET}"
[[ -f "${VALID_CSV}" ]] && echo -e "${BOLD}valid:${RESET}  ${YELLOW}${VALID_CSV}${RESET}" || true
echo -e "${BOLD}out:${RESET}    ${YELLOW}${OUT_DIR}${RESET}\n"

# ===== cronómetro =====
START_TIME=$(date +%s)

# ===== run =====
PYTHONPATH="${SRC_DIR}" \
python -m train \
  --train "${TRAIN_CSV}" \
  --out   "${OUT_DIR}" \
  --cfg   "${CFG}" \
  $( [[ -f "${VALID_CSV}" ]] && echo --valid "${VALID_CSV}" )

# ===== fin =====
END_TIME=$(date +%s); DURATION=$((END_TIME - START_TIME))
echo -e "\n${GREEN}${BOLD}✅ Entrenamiento completado${RESET}"
echo -e "${BOLD}Modelo:${RESET} ${YELLOW}${OUT_DIR}${RESET}"
echo -e "${BOLD}Duración:${RESET} ${YELLOW}${DURATION}s${RESET}\n"
