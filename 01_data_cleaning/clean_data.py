import os
import re
import json
import time
import hashlib
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split, GroupShuffleSplit

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
CACHE_KEY = "CleaningDone"  # misma key; puedes forzar con CLEAN_FORCE=1

# stopwords opcional solo para prompt
try:
    import nltk
    nltk.data.find("corpora/stopwords")
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

# ---------- utils ----------
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

def cleanText(text: str, remove_stopwords: bool, keep_punct: bool) -> str:
    # limpieza controlada: no sobre-limpiar respuestas
    if not isinstance(text, str):
        return None
    txt = text.lower()
    # quita urls/handles/hashtags
    txt = re.sub(r"(https?://\S+|www\.\S+)", " ", txt)
    txt = re.sub(r"@\w+", " ", txt)
    txt = re.sub(r"#\w+", " ", txt)
    # normaliza puntuación
    if keep_punct:
        # conserva . , ! ? ; :
        txt = re.sub(r"[^\w\s\.\,\!\?\;\:]", " ", txt)
    else:
        txt = re.sub(r"[^\w\s]", " ", txt)
    # colapsa espacios
    txt = re.sub(r"\s+", " ", txt).strip()
    # stopwords opcional (solo prompt)
    if remove_stopwords and STOPWORDS:
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

def hashPrompt(s: str) -> str:
    # hash de prompt para agrupar en split
    s = (s or "").strip().lower()
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def jaccardTokens(a: str, b: str) -> float:
    sa, sb = set(str(a).split()), set(str(b).split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)

def cosineTfidf_pairs(a_list, b_list):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vect = TfidfVectorizer(min_df=1)
    X = vect.fit_transform(list(a_list) + list(b_list))
    XA = X[:len(a_list)]
    XB = X[len(a_list):]
    num = (XA.multiply(XB)).sum(axis=1).A1
    den = (np.sqrt((XA.multiply(XA)).sum(axis=1).A1) * np.sqrt((XB.multiply(XB)).sum(axis=1).A1) + 1e-12)
    return (num / den).tolist()

# ---------- pipeline ----------
def cleanDataFrame(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    # limpieza menos agresiva en respuestas
    info = {"steps": [], "before_rows": int(len(df))}

    df["prompt_clean"] = df["prompt"].apply(evalAndJoin).apply(lambda x: cleanText(x, remove_stopwords=True, keep_punct=True))
    df["response_a_clean"] = df["response_a"].apply(evalAndJoin).apply(lambda x: cleanText(x, remove_stopwords=False, keep_punct=True))
    df["response_b_clean"] = df["response_b"].apply(evalAndJoin).apply(lambda x: cleanText(x, remove_stopwords=False, keep_punct=True))
    info["steps"].append({"step": "text_clean_controlled", "keep_punct": ".,!?;:", "stopwords_prompt": True, "stopwords_responses": False})

    # marcador "no response" solo si realmente vacío
    for col in ["response_a_clean", "response_b_clean"]:
        s = df[col].fillna("").str.strip()
        mask_na = s.str.fullmatch(r"(n/?a|na|none|no\s*answer|no\s*response)", case=False, na=False) | (s == "")
        s = s.mask(mask_na, "no response")
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

def dedupRows(df: pd.DataFrame, hard_cos=0.995, hard_jac=0.9, near_cos=0.98):
    # deduplicación por similitud A vs B y pares repetidos exactos
    a = df["response_a_clean"].fillna("")
    b = df["response_b_clean"].fillna("")
    cos = cosineTfidf_pairs(a.tolist(), b.tolist())
    jac = [jaccardTokens(x, y) for x, y in zip(a.tolist(), b.tolist())]
    df = df.copy()
    df["cosine_tfidf"] = cos
    df["jaccard_tokens"] = jac

    # pares casi idénticos A vs B en la MISMA fila
    df["hard_dup_pair"] = (df["cosine_tfidf"] >= hard_cos) & (df["jaccard_tokens"] >= hard_jac)
    df["near_dup_pair"] = (~df["hard_dup_pair"]) & (df["cosine_tfidf"] >= near_cos)

    # pares repetidos exactos en el dataset (misma A/B)
    def pairSig(x, y):
        return hashlib.md5((str(x) + "§" + str(y)).encode("utf-8")).hexdigest()
    df["pair_sig"] = [pairSig(x, y) for x, y in zip(a, b)]
    df["pair_dup_count"] = df.groupby("pair_sig")["pair_sig"].transform("count")

    # elimina duplicados duros
    before = len(df)
    keep_mask = ~df["hard_dup_pair"]
    df_pruned = df.loc[keep_mask].copy()
    removed = before - len(df_pruned)

    meta = {
        "hard_thresholds": {"cosine": hard_cos, "jaccard": hard_jac},
        "near_threshold": near_cos,
        "removed_hard_dups": int(removed),
        "kept_near_dups": int(df_pruned["near_dup_pair"].sum()),
        "pair_dup_over_1": int((df_pruned["pair_dup_count"] > 1).sum())
    }
    return df_pruned, meta

def makeSwappedDf(df: pd.DataFrame) -> pd.DataFrame:
    # duplica filas invirtiendo A/B y ajusta label, marca is_swapped
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

def stratifiedGroupSplit(df: pd.DataFrame, test_size=0.1, random_state=42):
    # intenta estratificar por label sin fuga de prompt; fallback a GroupShuffleSplit
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
        groups = df["prompt_hash"]
        X = df.index.values
        y = df["label"].values
        # toma el primer split como train/valid
        for train_idx, valid_idx in sgkf.split(X, y, groups):
            return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy(), {"splitter": "StratifiedGroupKFold"}
    except Exception:
        pass
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["prompt_hash"]
    X = df.index.values
    for train_idx, valid_idx in gss.split(X, groups=groups):
        return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy(), {"splitter": "GroupShuffleSplit"}

def runCleaning(input_filename: str = "train.csv"):
    # cache con bypass opcional
    if cache_manager.exists(CACHE_KEY) and os.environ.get("CLEAN_FORCE", "0") != "1":
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
    df_clean = generateLabel(df_clean).reset_index(drop=True)

    # longitudes para inspección rápida
    for col, newc in [("prompt_clean","prompt_len"),("response_a_clean","respA_len"),("response_b_clean","respB_len")]:
        df_clean[newc] = df_clean[col].str.split().str.len()

    # base limpia antes de dedup/split
    base_path = os.path.join(clean_dir, "data_clean_base.csv")
    df_clean.to_csv(base_path, index=False, encoding="utf-8-sig")

    print("deduplicando (A vs B y pares repetidos)...")
    df_dedup, dedup_meta = dedupRows(df_clean, hard_cos=0.995, hard_jac=0.9, near_cos=0.98)
    dedup_path = os.path.join(clean_dir, "data_clean_dedup.csv")
    df_dedup.to_csv(dedup_path, index=False, encoding="utf-8-sig")
    print(f"filas tras dedup: {len(df_dedup)} (removidos duros: {dedup_meta['removed_hard_dups']})")

    # hashing de prompt para split por grupos
    df_dedup["prompt_hash"] = df_dedup["prompt_clean"].map(hashPrompt)

    print("haciendo split por grupos (prompt) sin fuga...")
    train_base, valid_base, split_meta = stratifiedGroupSplit(df_dedup, test_size=0.1, random_state=42)
    # guarda bases sin swap
    train_base_path = os.path.join(clean_dir, "train_base.csv")
    valid_base_path = os.path.join(clean_dir, "valid_base.csv")
    train_base.to_csv(train_base_path, index=False, encoding="utf-8-sig")
    valid_base.to_csv(valid_base_path, index=False, encoding="utf-8-sig")

    # swap solo en train
    print("aplicando swap solo en train...")
    train_strat = makeSwappedDf(train_base)
    valid_strat = valid_base.copy()
    train_strat_path = os.path.join(clean_dir, "train_strat.csv")
    valid_strat_path = os.path.join(clean_dir, "valid_strat.csv")
    train_strat.to_csv(train_strat_path, index=False, encoding="utf-8-sig")
    valid_strat.to_csv(valid_strat_path, index=False, encoding="utf-8-sig")

    # meta y métricas
    def classPct(df):
        tot = max(len(df), 1)
        return df["label"].value_counts(normalize=True).sort_index().mul(100).round(2).to_dict()

    split_meta_out = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": input_filename,
        "rows_base": int(len(df_clean)),
        "rows_dedup": int(len(df_dedup)),
        "removed_hard_dups": dedup_meta["removed_hard_dups"],
        "kept_near_dups": dedup_meta["kept_near_dups"],
        "pair_dup_over_1": dedup_meta["pair_dup_over_1"],
        "splitter": split_meta.get("splitter", "unknown"),
        "train_rows_base": int(len(train_base)),
        "valid_rows_base": int(len(valid_base)),
        "train_rows_strat": int(len(train_strat)),
        "valid_rows_strat": int(len(valid_strat)),
        "class_pct_train_base": classPct(train_base),
        "class_pct_valid_base": classPct(valid_base),
        "class_pct_train_strat": classPct(train_strat),
        "class_pct_valid_strat": classPct(valid_strat),
        "is_swapped_train_pct": round(100.0 * (train_strat.get("is_swapped", pd.Series([0]*len(train_strat))).mean()), 2),
        "thresholds": {"hard_cos": 0.995, "hard_jac": 0.9, "near_cos": 0.98}
    }
    with open(os.path.join(clean_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(split_meta_out, f, ensure_ascii=False, indent=2)

    # artefactos legacy para compat con otros scripts
    df_dedup.to_csv(os.path.join(clean_dir, "data_clean.csv"), index=False, encoding="utf-8-sig")

    cache_manager.create(CACHE_KEY, content="data cleaned successfully")
    print("DONE")

# utilidad opcional para limpiar test con nueva lógica
def runCleaningTest(input_filename: str = "test.csv"):
    input_path = os.path.join(raw_dir, input_filename)
    if not os.path.exists(input_path):
        return {"test_clean_rows": 0}
    df_raw = pd.read_csv(input_path)
    df_raw["prompt_clean"] = df_raw["prompt"].apply(evalAndJoin).apply(lambda x: cleanText(x, True, True))
    df_raw["response_a_clean"] = df_raw["response_a"].apply(evalAndJoin).apply(lambda x: cleanText(x, False, True)).fillna("no response")
    df_raw["response_b_clean"] = df_raw["response_b"].apply(evalAndJoin).apply(lambda x: cleanText(x, False, True)).fillna("no response")
    df_raw = df_raw.dropna(subset=["prompt_clean", "response_a_clean", "response_b_clean"])
    df_raw = df_raw[
        (df_raw["prompt_clean"].str.strip() != "") &
        (df_raw["response_a_clean"].str.strip() != "") &
        (df_raw["response_b_clean"].str.strip() != "")
    ].copy()
    out = os.path.join(clean_dir, "test_clean.csv")
    df_raw.to_csv(out, index=False, encoding="utf-8-sig")
    return {"test_clean_rows": int(len(df_raw))}

if __name__ == "__main__":
    runCleaning()
