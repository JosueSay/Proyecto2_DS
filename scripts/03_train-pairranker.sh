#!/usr/bin/env bash
set -euo pipefail

# colores
GREEN="\033[1;32m"; BLUE="\033[1;34m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; RESET="\033[0m"; BOLD="\033[1m"

# rutas principales
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_DIR}/venv/bin/activate"
SRC_DIR="${REPO_DIR}/m_pair-ranker-deberta/src"
CFG="${REPO_DIR}/m_pair-ranker-deberta/configs/default.yaml"
TRAIN_CSV="${REPO_DIR}/data/clean/data_clean_aug.csv"
VALID_CSV="${REPO_DIR}/data/clean/valid_strat.csv"
OUT_DIR="${REPO_DIR}/results/deberta/run_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}" "${REPO_DIR}/reports/deberta"

# manejo de errores
ok_run=false
on_error() {
  local ec=$?
  echo -e "\n${RED}${BOLD}❌ Falló el entrenamiento (exit ${ec})${RESET}\n"
  exit "${ec}"
}
trap on_error ERR

# finalización exitosa
finish() {
  if "$ok_run"; then
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    echo -e "\n${GREEN}${BOLD}✅ Entrenamiento completado${RESET}"
    echo -e "${BOLD}Modelo:${RESET} ${YELLOW}${OUT_DIR}${RESET}"
    echo -e "${BOLD}Duración:${RESET} ${YELLOW}${DURATION}s${RESET}\n"
  fi
}
trap finish EXIT

# activar entorno virtual y variables
source "${VENV}"
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HOME="${REPO_DIR}/.hf_cache"

# chequeo rápido de dependencias
python - <<'PY'
try:
    import google.protobuf
    dep="ok"
except Exception as e:
    dep=f"falta protobuf: {e}"
print("protobuf:", dep)
PY

# banner
echo -e "\n${BLUE}${BOLD}🚀 Entrenamiento PairRanker (DeBERTa)${RESET}\n"

# info gpu/cpu
python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | BF16:", torch.cuda.is_bf16_supported())
else:
    print("CPU (GPU no disponible)")
PY

# resumen cfg
python - <<PY
import yaml, sys
cfg_path = "${CFG}"
try:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    print("pretrained_name:", cfg.get("pretrained_name"))
    print("batch_size:", cfg.get("batch_size"), "| epochs:", cfg.get("epochs"))
except Exception as e:
    print("no se pudo leer config:", e, file=sys.stderr)
    sys.exit(1)
PY

# resumen rutas
echo -e "${BOLD}config:${RESET} ${YELLOW}${CFG}${RESET}"
echo -e "${BOLD}train:${RESET}  ${YELLOW}${TRAIN_CSV}${RESET}"
[[ -f "${VALID_CSV}" ]] && echo -e "${BOLD}valid:${RESET}  ${YELLOW}${VALID_CSV}${RESET}" || true
echo -e "${BOLD}out:${RESET}    ${YELLOW}${OUT_DIR}${RESET}\n"

# cronómetro
START_TIME=$(date +%s)

# args para python
ARGS=( -m train --train "${TRAIN_CSV}" --out "${OUT_DIR}" --cfg "${CFG}" )
[[ -f "${VALID_CSV}" ]] && ARGS+=( --valid "${VALID_CSV}" )

# ejecutar entrenamiento
PYTHONPATH="${SRC_DIR}" python "${ARGS[@]}"

ok_run=true
