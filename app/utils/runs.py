import re
from pathlib import Path
from datetime import datetime
from .paths import modelResultsDir, modelReportsDir, listDirs

RUN_REGEX = re.compile(r"run_(\d{8}_\d{6})$")

# timestamp str -> datetime
def parseRunTimestamp(name):
    m = RUN_REGEX.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")

# descubre runs de un modelo y empareja results/reports
def findRunsForModel(model, cfg):
    res_root = modelResultsDir(model, cfg)
    rep_root = modelReportsDir(model, cfg)
    runs = []
    for d in listDirs(res_root):
        if not RUN_REGEX.match(d.name):
            continue
        ts = parseRunTimestamp(d.name)
        rep_dir = rep_root.joinpath(d.name)
        runs.append({
            "model": model,
            "name": d.name,
            "timestamp": ts,
            "results_dir": d,
            "reports_dir": rep_dir if rep_dir.exists() else None,
        })
    runs.sort(key=lambda r: (r["timestamp"] or datetime.min), reverse=True)
    return runs

# último run de un modelo (o None)
def getLatestRun(model, cfg):
    runs = findRunsForModel(model, cfg)
    return runs[0] if runs else None

# modelos habilitados con presencia de carpetas
def enabledModels(cfg):
    ms = []
    for m, meta in cfg["models"].items():
        if not meta.get("enabled", False):
            continue
        ms.append(m)
    return ms

# catálogo completo: modelo -> runs[]
def catalogAllRuns(cfg):
    cat = {}
    for m in enabledModels(cfg):
        cat[m] = findRunsForModel(m, cfg)
    return cat

# busca run por nombre exacto
def getRunByName(model, run_name, cfg):
    for r in findRunsForModel(model, cfg):
        if r["name"] == run_name:
            return r
    return None

# sanity: existe modelo y al menos un run
def modelReady(model, cfg):
    latest = getLatestRun(model, cfg)
    return latest is not None and latest["results_dir"].exists()
