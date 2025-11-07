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
from cache_manager import CacheManager  # noqa: E402

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
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))


def evalAndJoin(text: str):
    # parsea strings tipo lista y une por salto de línea
    if pd.isna(text):
        return None
    t = str(text).strip()
    if t.lower() in {"", "nan", "null", "none"}:
        return None
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
    # limpieza ligera
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
    
    for col in ["response_a_clean", "response_b_clean"]:
        s = df[col].fillna("")
        # normaliza marcadores de no-respuesta
        s = s.str.strip()
        mask_na = s.str.fullmatch(r"(n/?a|na|none|no\s*answer|no\s*response)", case=False, na=False)
        mask_empty = (s == "")
        s = s.mask(mask_na | mask_empty, "no response")
        df[col] = s

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

def makeSwappedDf(df: pd.DataFrame) -> pd.DataFrame:
    # duplica filas invirtiendo A/B y ajusta label
    swapped = df.copy()
    swapped["response_a_clean"], swapped["response_b_clean"] = df["response_b_clean"], df["response_a_clean"]
    if "model_a" in df.columns and "model_b" in df.columns:
        swapped["model_a"], swapped["model_b"] = df["model_b"], df["model_a"]
    map_lbl = {0: 1, 1: 0, 2: 2}
    swapped["label"] = swapped["label"].map(map_lbl)
    df = df.copy()
    df["is_swapped"] = 0
    swapped["is_swapped"] = 1
    return pd.concat([df, swapped], axis=0, ignore_index=True)

def runCleaningTest(input_filename: str = "test.csv"):
    # limpia test.csv con el mismo pipeline
    input_path = os.path.join(raw_dir, input_filename)
    if not os.path.exists(input_path):
        return {"test_clean_rows": 0}

    df_raw = pd.read_csv(input_path)
    df_raw["prompt_clean"] = df_raw["prompt"].apply(evalAndJoin).apply(cleanText)
    df_raw["response_a_clean"] = df_raw["response_a"].apply(evalAndJoin).apply(cleanText).fillna("no response")
    df_raw["response_b_clean"] = df_raw["response_b"].apply(evalAndJoin).apply(cleanText).fillna("no response")

    df_raw = df_raw.dropna(subset=["prompt_clean", "response_a_clean", "response_b_clean"])
    df_raw = df_raw[
        (df_raw["prompt_clean"].str.strip() != "") &
        (df_raw["response_a_clean"].str.strip() != "") &
        (df_raw["response_b_clean"].str.strip() != "")
    ].copy()

    out = os.path.join(clean_dir, "test_clean.csv")
    df_raw.to_csv(out, index=False, encoding="utf-8-sig")
    return {"test_clean_rows": int(len(df_raw))}

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

    # imputar único nan en response_a_clean
    imputations = int(df_clean["response_a_clean"].isna().sum())
    if imputations > 0:
        df_clean["response_a_clean"] = df_clean["response_a_clean"].fillna("no response")

    # longitudes para fijar max_len
    df_clean["prompt_len"] = df_clean["prompt_clean"].str.split().str.len()
    df_clean["respA_len"] = df_clean["response_a_clean"].str.split().str.len()
    df_clean["respB_len"] = df_clean["response_b_clean"].str.split().str.len()
    df_clean[["prompt_len", "respA_len", "respB_len"]].describe().to_csv(
        os.path.join(clean_dir, "length_stats.csv"), index=True
    )

    # anti-sesgo por posición: swap A↔B
    df_aug = makeSwappedDf(df_clean)
    df_aug.to_csv(os.path.join(clean_dir, "data_clean_aug.csv"), index=False, encoding="utf-8-sig")

    # split estratificado
    try:
        from sklearn.model_selection import train_test_split
        train_df, valid_df = train_test_split(
            df_aug, test_size=0.1, random_state=42, stratify=df_aug["label"]
        )
        train_df.to_csv(os.path.join(clean_dir, "train_strat.csv"), index=False, encoding="utf-8-sig")
        valid_df.to_csv(os.path.join(clean_dir, "valid_strat.csv"), index=False, encoding="utf-8-sig")
        split_info = {"train_rows": int(len(train_df)), "valid_rows": int(len(valid_df))}
    except Exception as e:
        split_info = {"split_error": str(e)}

    # limpia test.csv
    test_info = runCleaningTest("test.csv")

    # guarda resultados base (sin swap) si lo quieres como referencia
    data_clean_path = os.path.join(clean_dir, "data_clean.csv")
    process_info_path = os.path.join(clean_dir, "data_process_info.csv")

    proc_info["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    proc_info["source_file"] = input_filename
    proc_info["columns"] = list(df_clean.columns)
    proc_info["imputations_response_a_clean"] = imputations
    proc_info["augmented_rows"] = int(len(df_aug))
    proc_info.update(split_info)
    proc_info.update(test_info)

    df_clean.to_csv(data_clean_path, index=False, encoding="utf-8-sig")

    rows = []
    for k, v in proc_info.items():
        rows.append({"key": k, "value": json.dumps(v) if not isinstance(v, (str, int, float)) else v})
    pd.DataFrame(rows).to_csv(process_info_path, index=False, encoding="utf-8-sig")

    cache_manager.create(CACHE_KEY, content="data cleaned successfully")
    print("DONE")

if __name__ == "__main__":
    runCleaning()
