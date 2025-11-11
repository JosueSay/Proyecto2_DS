import os
from typing import List, Dict, Optional

# rutas base del proyecto según el árbol compartido
def projectRoot() -> str:
    # asume ejecución desde raíz del repo; ajusta si lo corres desde otro cwd
    return os.path.abspath(".")

def reportsRoot() -> str:
    return os.path.join(projectRoot(), "reports")

def imagesResultsDir() -> str:
    return os.path.join(projectRoot(), "images", "resultados")

def listAvailableModels() -> List[str]:
    # basado en estructura del repo
    return ["deberta", "roberta", "electra", "xlnet"]

def listRuns(model: str) -> List[str]:
    base = os.path.join(reportsRoot(), model)
    if not os.path.isdir(base):
        return []
    # solo carpetas tipo run_YYYYMMDD_HHMMSS
    runs = [d for d in os.listdir(base) if d.startswith("run_") and os.path.isdir(os.path.join(base, d))]
    runs.sort()  # lexicográfico funciona por el timestamp en nombre
    return runs

def latestRunDir(model: str) -> Optional[str]:
    runs = listRuns(model)
    if not runs:
        return None
    return os.path.join(reportsRoot(), model, runs[-1])

def expectedCsvs(run_dir: str) -> Dict[str, str]:
    # csv esenciales generados por m_pair-ranker
    return {
        "epochs": os.path.join(run_dir, "epochs.csv"),
        "class_report": os.path.join(run_dir, "class_report.csv"),
        "confusion": os.path.join(run_dir, "confusion.csv"),
        "pred_distributions": os.path.join(run_dir, "pred_distributions.csv"),
        "steps": os.path.join(run_dir, "steps.csv"),
    }
