import os
import sys
import pandas as pd

# agregar ruta del repo para reutilizar pipeline existente
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PIPE_ROOT = os.path.join(REPO_ROOT, "01_data_cleaning", "core")
if PIPE_ROOT not in sys.path:
    sys.path.append(PIPE_ROOT)

from text_utils import evalAndJoin, cleanText
from truncation import applyTailControl

# columnas fuente y destino
SRC_COLS = ["id", "prompt", "response_a", "response_b"]
DST_COLS = ["id", "prompt_clean", "response_a_clean", "response_b_clean"]

def _ensureCols(df: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in SRC_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"faltan columnas en df: {miss}")
    return df[SRC_COLS].copy()

def _basicCleanText(s: str) -> str:
    # min-limpieza: urls/mentions/punct, sin stopwords
    return cleanText(evalAndJoin(s) or "", remove_stopwords=False, keep_punct=True) or ""

def cleanBatch(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensureCols(df)
    out = pd.DataFrame()
    out["id"] = df["id"]
    out["prompt_clean"] = df["prompt"].map(_basicCleanText)
    out["response_a_clean"] = df["response_a"].map(_basicCleanText)
    out["response_b_clean"] = df["response_b"].map(_basicCleanText)
    out = applyTailControl(out)  # controla longitudes vs presupuesto del modelo
    return out[DST_COLS]

def cleanSingle(prompt: str, resp_a: str, resp_b: str, row_id: int | str = "single") -> pd.DataFrame:
    tmp = pd.DataFrame([{"id": row_id, "prompt": prompt, "response_a": resp_a, "response_b": resp_b}])
    return cleanBatch(tmp)

def cleanCsvPath(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return cleanBatch(df)
