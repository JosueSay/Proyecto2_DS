#!/usr/bin/env bash
set -euo pipefail

# colores
GREEN="\033[1;32m"; BLUE="\033[1;34m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; RESET="\033[0m"; BOLD="\033[1m"

# rutas base
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_DIR}/venv/bin/activate"
SRC_DIR="${REPO_DIR}/m_pair-ranker/src"
DEFAULT_CFG="${REPO_DIR}/m_pair-ranker/configs/deberta.yaml"

# permitir pasar cfg por argumento: ./01_train-pairranker.sh path/a/mi.yaml
CFG="${1:-$DEFAULT_CFG}"

# ── traps
ok_run=false
on_error() {
  local ec=$?
  echo -e "\n${RED}${BOLD}❌ Falló el entrenamiento (exit ${ec})${RESET}\n"
  exit "${ec}"
}
trap on_error ERR

finish() {
  if "$ok_run"; then
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    echo -e "\n${GREEN}${BOLD}✅ Entrenamiento completado${RESET}"
    echo -e "${BOLD}Duración:${RESET} ${YELLOW}${DURATION}s${RESET}\n"

    # localizar últimos run_* coherentes entre results y reports según el YAML
    python - <<PY
import os, sys, yaml, glob
cfg_path = "${CFG}"
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
runs_root = cfg["logging"]["runs_dir"]
reps_root = cfg["logging"]["reports_dir"]
def last_run(root):
    pats = sorted(glob.glob(os.path.join(root, "run_*")))
    return pats[-1] if pats else None
last_runs = last_run(runs_root), last_run(reps_root)
print("runs_dir:", last_runs[0] or "(no encontrado)")
print("report_dir:", last_runs[1] or "(no encontrado)")
PY
  fi
}
trap finish EXIT

# activar venv
if [[ -f "${VENV}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV}"
fi

eval "$(
python - "$CFG" <<'PY'
import yaml, os, sys
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
env = (cfg.get("env") or {})
def b2int(b):
    return "1" if str(b).lower() in {"1","true","yes","y"} else "0"
tok_par = env.get("tokenizers_parallelism", False)
clb     = env.get("cuda_launch_blocking", 0)
alloc   = env.get("pytorch_cuda_alloc_conf", "max_split_size_mb:128")
hf_home = env.get("hf_home", os.path.join(os.path.dirname(cfg_path), "..", ".hf_cache"))
print(f'export TOKENIZERS_PARALLELISM={str(tok_par).lower()}')
print(f'export CUDA_LAUNCH_BLOCKING={int(clb)}')
print(f'export PYTORCH_CUDA_ALLOC_CONF="{alloc}"')
print(f'export HF_HOME="{hf_home}"')
PY
)"


# chequeo rápido de deps
python - <<'PY'
try:
    import google.protobuf  # noqa
    print("protobuf: ok")
except Exception as e:
    print(f"protobuf: falta ({e})")
PY

# banner
echo -e "\n${BLUE}${BOLD}🚀 Entrenamiento PairRanker${RESET}\n"

# info gpu/cpu
python - <<'PY'
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | BF16:", torch.cuda.is_bf16_supported())
else:
    print("CPU (GPU no disponible)")
PY

# resumen cfg
python - "$CFG" <<'PY'
import yaml, sys
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    c = yaml.safe_load(f)
m = c.get("model", {})
t = c.get("train", {})
d = c.get("data", {})
log = c.get("logging", {})
print("pretrained_name:", m.get("pretrained_name"))
print("batch_size:", t.get("batch_size"), "| epochs:", t.get("epochs"))
print("train_csv:", d.get("train_csv"))
print("valid_csv:", d.get("valid_csv"))
print("runs_dir:", log.get("runs_dir"))
print("reports_dir:", log.get("reports_dir"))
PY

echo -e "${BOLD}config:${RESET} ${YELLOW}${CFG}${RESET}\n"

# cronómetro
START_TIME=$(date +%s)

# ejecutar entrenamiento (usa YAML para rutas y dirs)
PYTHONPATH="${SRC_DIR}" python -m pairranker.train.loop --cfg "${CFG}"

ok_run=true
