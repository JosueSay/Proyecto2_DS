import os
import re
import json
import time
import pandas as pd
from typing import Tuple
from nltk.corpus import stopwords

# deps locales
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "00_cache-manager"))
from cache_manager import CacheManager 

# rutas base
project_dir = os.path.join(BASE_DIR, "..")
data_dir = os.path.join(project_dir, "data")
raw_dir = data_dir
clean_dir = os.path.join(data_dir, "clean")
cache_dir = os.path.join(project_dir, "cache")
os.makedirs(clean_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# cache
cache_config = os.path.join(project_dir, "00_cache-manager", "cache_config.yaml")
cache_manager = CacheManager(cache_config, cache_dir)
CACHE_KEY = "CleaningDone"

# stopwords
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()
    
    
def evalAndJoin(text: str):
    # convierte strings que representan arrays a texto unido por saltos de línea
    if pd.isna(text):
        return None
    t = str(text).strip()
    if t.lower() in {"", "nan", "null", "none"}:
        return None
    # normaliza escapes frecuentes
    t = t.replace(r"\/", "/").replace(r"null", '""')
    try:
        arr = eval(t)
        if isinstance(arr, list):
            arr = [str(a).strip() for a in arr if str(a).strip()]
            return "\n".join(arr) if arr else None
        if arr is None:
            return None
        return str(arr)
    except Exception:
        return t or None

def cleanText(text: str) -> str:
    # limpieza ligera para prompts/respuestas
    if not isinstance(text, str):
        return None
    txt = text.lower()
    txt = re.sub(r"(https?://\S+|www\.\S+)", " ", txt)
    txt = re.sub(r"@\w+", " ", txt)
    txt = re.sub(r"#\w+", " ", txt)
    txt = re.sub(r"[_\*\~\^\(\)\[\]\{\}\|]", " ", txt)
    txt = re.sub(r"\.", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    if STOPWORDS:
        txt = " ".join(w for w in txt.split() if w not in STOPWORDS).strip()
    return txt or None

def removeInvalidChars(text: str) -> str:
    # evita errores de codificación
    return text.encode("utf-8", errors="replace").decode("utf-8")

def generateLabel(df: pd.DataFrame) -> pd.DataFrame:
    # label: 0 gana A, 1 gana B, 2 empate
    def mapLabel(row):
        tie = int(row.get("winner_tie", 0))
        a = int(row.get("winner_model_a", 0))
        b = int(row.get("winner_model_b", 0))
        if tie + a + b != 1:
            return pd.NA
        if tie == 1:
            return 2
        return 0 if a == 1 else 1
    df["label"] = df.apply(mapLabel, axis=1)
    return df[df["label"].notna()].copy()

def cleanDataFrame(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    # aplica pipeline de limpieza y devuelve df limpio + info de proceso
    info = {"steps": [], "before_rows": int(len(df))}
    info["steps"].append({"step": "eval_join", "cols": ["prompt", "response_a", "response_b"]})

    df["prompt_clean"] = df["prompt"].apply(evalAndJoin)
    df["response_a_clean"] = df["response_a"].apply(evalAndJoin)
    df["response_b_clean"] = df["response_b"].apply(evalAndJoin)

    info["steps"].append({"step": "text_clean", "cols": ["prompt_clean", "response_a_clean", "response_b_clean"]})
    df["prompt_clean"] = df["prompt_clean"].apply(cleanText)
    df["response_a_clean"] = df["response_a_clean"].apply(cleanText)
    df["response_b_clean"] = df["response_b_clean"].apply(cleanText)

    # descarta vacíos
    before_drop = int(len(df))
    df = df.dropna(subset=["prompt_clean", "response_a_clean", "response_b_clean"]).copy()
    df = df[
        (df["prompt_clean"].str.strip() != "") &
        (df["response_a_clean"].str.strip() != "") &
        (df["response_b_clean"].str.strip() != "")
    ].copy()
    info["steps"].append({"step": "drop_empty", "dropped_rows": before_drop - int(len(df))})

    # sanea codificación
    for col in ["prompt_clean", "response_a_clean", "response_b_clean"]:
        df[col] = df[col].apply(removeInvalidChars)

    info["after_rows"] = int(len(df))
    return df, info



def runCleaning(input_filename: str = "train.csv"):
    # usa cache si ya se limpió
    if cache_manager.exists(CACHE_KEY):
        print("CACHE_USED")
        return

    input_path = os.path.join(raw_dir, input_filename)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"no se encontró el archivo: {input_path}")

    print("cargando dataset original...")
    df_raw = pd.read_csv(input_path)
    print(f"registros originales: {len(df_raw)}")

    print("limpiando datos...")
    df_clean, proc_info = cleanDataFrame(df_raw)
    print(f"registros post-limpieza: {len(df_clean)}")

    print("generando labels...")
    df_clean = generateLabel(df_clean)

    # guarda resultados requeridos
    data_clean_path = os.path.join(clean_dir, "data_clean.csv")
    process_info_path = os.path.join(clean_dir, "data_process_info.csv")

    # agrega metadatos mínimos
    proc_info["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    proc_info["source_file"] = input_filename
    proc_info["columns"] = list(df_clean.columns)

    # data_clean
    df_clean.to_csv(data_clean_path, index=False, encoding="utf-8-sig")

    # data_process_info: cada fila = key, value
    rows = []
    for k, v in proc_info.items():
        rows.append({"key": k, "value": json.dumps(v) if not isinstance(v, (str, int, float)) else v})
    pd.DataFrame(rows).to_csv(process_info_path, index=False, encoding="utf-8-sig")

    # crea cache
    cache_manager.create(CACHE_KEY, content="data cleaned successfully")
    print("DONE")

if __name__ == "__main__":
    runCleaning()
        