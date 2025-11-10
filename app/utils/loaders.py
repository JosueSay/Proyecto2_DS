from pathlib import Path
import pandas as pd
import yaml
import streamlit as st
from .paths import repoPath
from .runs import getLatestRun, getRunByName
from .state import getAppConfig

# lee csv genérico con cache
@st.cache_data(show_spinner=False)
def loadCsv(csv_path: str):
    p = Path(csv_path)
    if not p.exists():
        return None
    return pd.read_csv(p)

# lee csv de reports para un run
@st.cache_data(show_spinner=False)
def loadReportCsv(model: str, run_name: str, filename: str):
    cfg = getAppConfig()
    run = getRunByName(model, run_name, cfg) if run_name else getLatestRun(model, cfg)
    if not run or not run.get("reports_dir"):
        return None
    csv_path = Path(run["reports_dir"]) / filename
    return loadCsv(str(csv_path))

# lee config del run (prioriza reports/run_config_used.yaml; fallback results/train_config.yaml)
@st.cache_data(show_spinner=False)
def loadRunConfig(model: str, run_name: str):
    cfg = getAppConfig()
    run = getRunByName(model, run_name, cfg) if run_name else getLatestRun(model, cfg)
    if not run:
        return None
    cand = [
        Path(run["reports_dir"] or "") / "run_config_used.yaml",
        Path(run["results_dir"]) / "train_config.yaml",
        Path(run["results_dir"]) / "config.json",
    ]
    for p in cand:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                try:
                    return yaml.safe_load(f)
                except Exception:
                    try:
                        import json
                        return json.load(f)
                    except Exception:
                        return None
    return None

# paths a artefactos del modelo (no carga a memoria)
def getArtifacts(model: str, run_name: str):
    cfg = getAppConfig()
    run = getRunByName(model, run_name, cfg) if run_name else getLatestRun(model, cfg)
    if not run:
        return None
    res_dir = Path(run["results_dir"])
    tok_dir = res_dir / "tokenizer"
    model_bin = res_dir / "model.bin"
    cfg_train = res_dir / "train_config.yaml"
    return {
        "model": model,
        "run_name": run["name"],
        "results_dir": res_dir,
        "reports_dir": Path(run["reports_dir"]) if run.get("reports_dir") else None,
        "tokenizer_dir": tok_dir if tok_dir.exists() else None,
        "model_bin": model_bin if model_bin.exists() else None,
        "train_config": cfg_train if cfg_train.exists() else None,
    }

# helpers rápidos para csvs típicos
def loadEpochs(model: str, run_name: str):
    cfg = getAppConfig()
    fname = cfg["reports_expected"]["epochs"]
    return loadReportCsv(model, run_name, fname)

def loadSteps(model: str, run_name: str):
    cfg = getAppConfig()
    fname = cfg["reports_expected"]["steps"]
    return loadReportCsv(model, run_name, fname)

def loadAlerts(model: str, run_name: str):
    cfg = getAppConfig()
    fname = cfg["reports_expected"]["alerts"]
    return loadReportCsv(model, run_name, fname)

def loadConfusion(model: str, run_name: str):
    cfg = getAppConfig()
    fname = cfg["reports_expected"]["confusion"]
    return loadReportCsv(model, run_name, fname)

def loadClassReport(model: str, run_name: str):
    cfg = getAppConfig()
    fname = cfg["reports_expected"]["class_report"]
    return loadReportCsv(model, run_name, fname)
