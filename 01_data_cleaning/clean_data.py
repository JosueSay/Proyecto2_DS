import os
import json
import time
import pandas as pd
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "00_cache-manager"))
from cache_manager import CacheManager

# dirs base
project_dir = os.path.join(BASE_DIR, "..")
data_dir = os.path.join(project_dir, "data")
raw_dir = data_dir
clean_dir = os.path.join(data_dir, "clean")
cache_dir = os.path.join(project_dir, "cache")
os.makedirs(clean_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# init cache
cache_config = os.path.join(project_dir, "00_cache-manager", "cache_config.yaml")
cache_manager = CacheManager(cache_config, cache_dir)
CACHE_KEY = "CleaningDone"

from core.settings import MAX_LEN_RESP
from core.text_utils import (
    evalAndJoin, cleanText, removeInvalidChars, promptSig,
)
from core.labels import generateLabel
from core.truncation import applyTailControl
from core.dedup import dedupRows, clusterByTemplate
from core.splitter import makeSwappedDf, stratifiedGroupSplit


def cleanDataFrame(df: pd.DataFrame):
    info = {"steps": [], "before_rows": int(len(df))}

    # limpiar prompts y responses
    df["prompt_clean"] = df["prompt"].apply(evalAndJoin).apply(
        lambda x: cleanText(x, remove_stopwords=True, keep_punct=True)
    )
    df["response_a_clean"] = df["response_a"].apply(evalAndJoin).apply(
        lambda x: cleanText(x, remove_stopwords=False, keep_punct=True)
    )
    df["response_b_clean"] = df["response_b"].apply(evalAndJoin).apply(
        lambda x: cleanText(x, remove_stopwords=False, keep_punct=True)
    )
    info["steps"].append({
        "step": "text_clean_controlled",
        "keep_punct": ".,!?;:",
        "stopwords_prompt": True,
        "stopwords_responses": False
    })

    # reemplazar n/a o vacíos por "no response"
    for col in ["response_a_clean", "response_b_clean"]:
        s = df[col].fillna("").str.strip()
        mask_na = s.str.fullmatch(r"(n/?a|na|none|no\s*answer|no\s*response)", case=False, na=False) | (s == "")
        s = s.mask(mask_na, "no response")
        df[col] = s

    # eliminar filas con campos vacíos
    before_drop = int(len(df))
    df = df.dropna(subset=["prompt_clean", "response_a_clean", "response_b_clean"]).copy()
    df = df[
        (df["prompt_clean"].str.strip() != "") &
        (df["response_a_clean"].str.strip() != "") &
        (df["response_b_clean"].str.strip() != "")
    ].copy()
    info["steps"].append({"step": "drop_empty", "dropped_rows": before_drop - int(len(df))})

    # limpiar chars inválidos
    for col in ["prompt_clean", "response_a_clean", "response_b_clean"]:
        df[col] = df[col].apply(removeInvalidChars)

    df["prompt_sig"] = df["prompt_clean"].map(promptSig)

    info["after_rows"] = int(len(df))
    return df, info


def runCleaning(input_filename: str = "train.csv"):
    # usar cache si existe y no forzamos limpieza
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
    df_clean, _ = cleanDataFrame(df_raw)
    print(f"registros post-limpieza: {len(df_clean)}")

    print("generando labels...")
    df_clean = generateLabel(df_clean).reset_index(drop=True)

    # stats de longitud de texto
    df_clean["prompt_len"] = df_clean["prompt_clean"].str.split().str.len()
    df_clean["respA_len"] = df_clean["response_a_clean"].str.split().str.len()
    df_clean["respB_len"] = df_clean["response_b_clean"].str.split().str.len()
    length_stats = df_clean[["prompt_len", "respA_len", "respB_len"]].describe(percentiles=[.99])
    length_stats.to_csv(os.path.join(clean_dir, "length_stats.csv"), index=True)

    # controlar colas largas
    p99_hint = {
        "respA_len_p99": float(length_stats.loc["99%", "respA_len"]) if "99%" in length_stats.index else df_clean["respA_len"].quantile(.99),
        "respB_len_p99": float(length_stats.loc["99%", "respB_len"]) if "99%" in length_stats.index else df_clean["respB_len"].quantile(.99),
    }
    df_clean = applyTailControl(df_clean, max_len_resp=MAX_LEN_RESP, p99_hint=p99_hint)

    # recalcular longitudes después de truncar
    df_clean["respA_len"] = df_clean["response_a_clean"].str.split().str.len()
    df_clean["respB_len"] = df_clean["response_b_clean"].str.split().str.len()

    df_clean.to_csv(os.path.join(clean_dir, "data_clean_base.csv"), index=False, encoding="utf-8-sig")

    # deduplicar agresivo
    print("deduplicando agresivo (A vs B, near-dups y anti-TIE trivial)...")
    df_dedup, dedup_meta = dedupRows(df_clean, hard_cos=0.998, hard_jac=0.94, near_cos=0.98)

    # anti-clonado por plantilla
    print("anti-clonado por plantilla y cap por prompt...")
    df_templ, templ_meta = clusterByTemplate(df_dedup, max_per_prompt=8, cap_start=10, sim_thr=0.985)

    dedup_path = os.path.join(clean_dir, "data_clean_dedup.csv")
    df_templ.to_csv(dedup_path, index=False, encoding="utf-8-sig")
    print(f"filas tras dedup+template: {len(df_templ)} (removidos aprox: {dedup_meta['removed_total']})")

    # split por grupos sin fuga
    print("haciendo split por grupos (prompt_sig) sin fuga...")
    train_base, valid_base, split_meta = stratifiedGroupSplit(df_templ, test_size=0.1, random_state=42)
    train_base.to_csv(os.path.join(clean_dir, "train_base.csv"), index=False, encoding="utf-8-sig")
    valid_base.to_csv(os.path.join(clean_dir, "valid_base.csv"), index=False, encoding="utf-8-sig")

    # swap solo en train
    print("aplicando swap solo en train...")
    train_strat = makeSwappedDf(train_base)
    valid_strat = valid_base.copy()
    train_strat.to_csv(os.path.join(clean_dir, "train_strat.csv"), index=False, encoding="utf-8-sig")
    valid_strat.to_csv(os.path.join(clean_dir, "valid_strat.csv"), index=False, encoding="utf-8-sig")

    # stats de prompts únicos y intersección
    train_prompts = set(train_base.get("prompt_sig", pd.Series(dtype=str)).astype(str).tolist())
    valid_prompts = set(valid_base.get("prompt_sig", pd.Series(dtype=str)).astype(str).tolist())
    inter = sorted(train_prompts & valid_prompts)
    pu_train = len(train_prompts)
    pu_valid = len(valid_prompts)
    inter_count = len(inter)
    inter_pct = (inter_count / max(1, min(pu_train, pu_valid))) * 100.0
    inter_examples = inter[:20]

    def classPct(df):
        tot = max(len(df), 1)
        return df["label"].value_counts(normalize=True).sort_index().mul(100).round(2).to_dict()

    # guardar metadata split
    split_meta_out = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": input_filename,
        "rows_base": int(len(df_clean)),
        "rows_dedup": int(len(df_templ)),
        "clusters_kept": templ_meta["clusters_kept"],
        "capped_prompts": templ_meta["capped_prompts"],
        "max_per_prompt": templ_meta["max_per_prompt"],
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
        "thresholds": {"hard_cos": 0.998, "hard_jac": 0.94, "near_cos": 0.98, "template_sim": 0.985},
        "p99": {"respA": float(p99_hint["respA_len_p99"]), "respB": float(p99_hint["respB_len_p99"])},
        "removed_total": dedup_meta["removed_total"],
        "removed_hard_dups": dedup_meta.get("removed_hard_dups"),
        "removed_near_dups": dedup_meta.get("removed_near_dups"),
        "kept_near_dups": dedup_meta.get("kept_near_dups"),
        "prompts_unique_train": pu_train,
        "prompts_unique_valid": pu_valid,
        "prompts_intersection_count": inter_count,
        "prompts_intersection_pct": round(inter_pct, 2),
        "prompts_intersection_examples": inter_examples,
    }
    with open(os.path.join(clean_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(split_meta_out, f, ensure_ascii=False, indent=2)

    # guardar csv final
    df_templ.to_csv(os.path.join(clean_dir, "data_clean.csv"), index=False, encoding="utf-8-sig")

    cache_manager.create(CACHE_KEY, content="data cleaned successfully")
    print("DONE")


def runCleaningTest(input_filename: str = "test.csv"):
    input_path = os.path.join(raw_dir, input_filename)
    if not os.path.exists(input_path):
        return {"test_clean_rows": 0}
    
    # limpieza rápida para test
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
