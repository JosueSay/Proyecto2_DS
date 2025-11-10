from pathlib import Path
import os
import yaml

# carga de yaml simple
def loadYaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# raíz del repo desde /app/utils/*
def repoRoot():
    return Path(__file__).resolve().parents[2]

# une rutas relativo a repo
def repoPath(*parts):
    return repoRoot().joinpath(*parts)

# asegura carpeta
def ensureDir(path_like):
    p = Path(path_like)
    p.mkdir(parents=True, exist_ok=True)
    return p

# paths base según config
def baseDirs(cfg):
    results_root = repoPath(cfg["paths"]["results_root"])
    reports_root = repoPath(cfg["paths"]["reports_root"])
    data_root = repoPath(cfg["paths"]["data_root"])
    images_root = repoPath(cfg["paths"]["images_root"])
    return dict(results_root=results_root, reports_root=reports_root, data_root=data_root, images_root=images_root)

# dir de resultados por modelo
def modelResultsDir(model, cfg):
    m = cfg["models"][model]
    return repoPath(m["results_dir"])

# dir de reports por modelo
def modelReportsDir(model, cfg):
    m = cfg["models"][model]
    return repoPath(m["reports_dir"])

# lista subcarpetas directas
def listDirs(path_like):
    p = Path(path_like)
    if not p.exists():
        return []
    return [d for d in p.iterdir() if d.is_dir()]

# comprueba existencia de un archivo dentro de una carpeta
def hasFile(dir_path, filename):
    return Path(dir_path).joinpath(filename).exists()
