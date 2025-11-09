import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# deps locales
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "00_cache-manager"))
from cache_manager import CacheManager

# rutas base
project_dir = os.path.join(BASE_DIR, "..")
clean_dir = os.path.join(project_dir, "data", "clean")
reports_dir = os.path.join(project_dir, "reports", "clean")
os.makedirs(reports_dir, exist_ok=True)
cache_dir = os.path.join(project_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)

# cache
cache_config = os.path.join(project_dir, "00_cache-manager", "cache_config.yaml")
cache_manager = CacheManager(cache_config, cache_dir)
CACHE_KEY = "SummaryDone"
CACHE_KEY_TV = "SummaryTrainValid"

# utils
def summarizeDf(df: pd.DataFrame) -> dict:
    dfs = {}

    dtypes_df = pd.DataFrame(df.dtypes, columns=["dtype"]).reset_index().rename(columns={"index": "column"})
    dfs["dtypes"] = dtypes_df

    n = len(df)
    na_counts = df.isna().sum()
    na_percent = (na_counts / max(n, 1) * 100).round(2)
    dfs["na_overview"] = pd.DataFrame({
        "column": na_counts.index,
        "na_count": na_counts.values,
        "na_percent": na_percent.values
    })

    dfs["head"] = df.head(10)
    dfs["tail"] = df.tail(10)

    try:
        dfs["describe"] = df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        dfs["describe"] = df.describe(include="all")

    try:
        dfs["describe_numeric"] = df.describe(numeric_only=True)
    except TypeError:
        num_cols = df.select_dtypes(include=["number"]).columns
        dfs["describe_numeric"] = df[num_cols].describe() if len(num_cols) else pd.DataFrame()

    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if dt_cols:
        rows = []
        for c in dt_cols:
            s = df[c].dropna()
            rows.append({
                "column": c,
                "min": s.min() if not s.empty else None,
                "max": s.max() if not s.empty else None,
                "nunique": s.nunique()
            })
        dfs["describe_datetime"] = pd.DataFrame(rows)

    for col in ["prompt_clean", "response_a_clean", "response_b_clean"]:
        if col in df.columns:
            mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            dfs[f"empty_{col}"] = df.loc[mask].head(50)

    return dfs

def saveReports(dfs: dict, prefix: str = ""):
    # guarda cada dataframe en reports/clean
    for name, d in dfs.items():
        out_name = f"{prefix}{name}.csv" if prefix else f"{name}.csv"
        out_path = os.path.join(reports_dir, out_name)
        try:
            d.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception:
            d.to_csv(out_path, encoding="utf-8-sig")

# helpers
def tokenizeLenSeries(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.split().map(len)

def classBalance(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    counts = df["label"].value_counts(dropna=False).sort_index()
    total = counts.sum()
    out = pd.DataFrame({
        "split": split_name,
        "label": counts.index.astype(int),
        "count": counts.values,
        "percent": (counts.values / max(total, 1) * 100).round(2)
    })
    return out

def lengthStatsByClass(df: pd.DataFrame) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "label": df["label"].astype(int),
        "prompt_len_tokens": tokenizeLenSeries(df["prompt_clean"]),
        "respA_len_tokens": tokenizeLenSeries(df["response_a_clean"]),
        "respB_len_tokens": tokenizeLenSeries(df["response_b_clean"])
    })
    agg = {}
    for col in ["prompt_len_tokens", "respA_len_tokens", "respB_len_tokens"]:
        agg[col] = ["mean", "std", "median", lambda x: x.quantile(0.9), lambda x: x.quantile(0.99)]
    out = tmp.groupby("label").agg(agg)
    out.columns = ["_".join([a for a in col if a]) for col in out.columns.to_flat_index()]
    out = out.reset_index().rename(columns={
        "prompt_len_tokens_<lambda_0>": "prompt_len_tokens_p90",
        "prompt_len_tokens_<lambda_1>": "prompt_len_tokens_p99",
        "respA_len_tokens_<lambda_0>": "respA_len_tokens_p90",
        "respA_len_tokens_<lambda_1>": "respA_len_tokens_p99",
        "respB_len_tokens_<lambda_0>": "respB_len_tokens_p90",
        "respB_len_tokens_<lambda_1>": "respB_len_tokens_p99",
        "median": "p50"
    })
    out = out.rename(columns={
        "prompt_len_tokens_median": "prompt_len_tokens_p50",
        "respA_len_tokens_median": "respA_len_tokens_p50",
        "respB_len_tokens_median": "respB_len_tokens_p50",
    })
    desired = [
        "label",
        "prompt_len_tokens_mean","prompt_len_tokens_std","prompt_len_tokens_p50","prompt_len_tokens_p90","prompt_len_tokens_p99",
        "respA_len_tokens_mean","respA_len_tokens_std","respA_len_tokens_p50","respA_len_tokens_p90","respA_len_tokens_p99",
        "respB_len_tokens_mean","respB_len_tokens_std","respB_len_tokens_p50","respB_len_tokens_p90","respB_len_tokens_p99",
    ]
    cols = [c for c in desired if c in out.columns]
    return out[cols]

def histoLengths(df: pd.DataFrame, col: str, bins: list, split_name: str) -> pd.DataFrame:
    lens = tokenizeLenSeries(df[col])
    labels = df["label"].astype(int).values
    rows = []
    counts, edges = None, None
    for lbl in sorted(np.unique(labels)):
        mask = labels == lbl
        counts, edges = np.histogram(lens[mask], bins=bins)
        for i in range(len(counts)):
            rows.append({
                "label": int(lbl),
                "bin_left": int(edges[i]),
                "bin_right": int(edges[i+1]),
                "count": int(counts[i]),
                "split": split_name
            })
    return pd.DataFrame(rows)

def computeSimilarity(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    a = df["response_a_clean"].fillna("").astype(str).tolist()
    b = df["response_b_clean"].fillna("").astype(str).tolist()

    def jaccard_pair(x, y):
        sx, sy = set(x.split()), set(y.split())
        if not sx and not sy:
            return 1.0
        inter = len(sx & sy)
        union = len(sx | sy)
        return inter / max(union, 1)

    jacc = [jaccard_pair(x, y) for x, y in zip(a, b)]

    vect = TfidfVectorizer(min_df=1)
    X = vect.fit_transform(a + b)
    XA = X[:len(a)]
    XB = X[len(a):]
    cos_diag = (XA.multiply(XB)).sum(axis=1).A1 / (
        np.sqrt((XA.multiply(XA)).sum(axis=1).A1) * np.sqrt((XB.multiply(XB)).sum(axis=1).A1) + 1e-12
    )

    abs_len_diff = (tokenizeLenSeries(df["response_a_clean"]) - tokenizeLenSeries(df["response_b_clean"])).abs()

    return pd.DataFrame({
        "jaccard_tokens": np.round(jacc, 6),
        "cosine_tfidf": np.round(cos_diag, 6),
        "abs_len_diff": abs_len_diff,
        "split": split_name
    })

def nearDuplicates(sim_df: pd.DataFrame, df_src: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    mask = (sim_df["jaccard_tokens"] >= threshold) | (sim_df["cosine_tfidf"] >= threshold)
    sub = df_src.loc[mask.values].copy()
    sub = sub.reset_index().rename(columns={"index": "row_id"})
    sub["jaccard_tokens"] = sim_df.loc[mask, "jaccard_tokens"].values
    sub["cosine_tfidf"] = sim_df.loc[mask, "cosine_tfidf"].values

    def shorten(t, n=160):
        t = str(t)
        return t if len(t) <= n else t[:n] + "..."
    sub["prompt_preview"] = sub["prompt_clean"].map(lambda x: shorten(x))
    sub["response_a_preview"] = sub["response_a_clean"].map(lambda x: shorten(x))
    sub["response_b_preview"] = sub["response_b_clean"].map(lambda x: shorten(x))
    return sub[[
        "row_id","label","jaccard_tokens","cosine_tfidf",
        "prompt_preview","response_a_preview","response_b_preview"
    ]]

def truncationImpact(df: pd.DataFrame, split_name: str, max_len_prompt: int = 512, max_len_resp: int = 512) -> pd.DataFrame:
    prompt_len = tokenizeLenSeries(df["prompt_clean"])
    a_len = tokenizeLenSeries(df["response_a_clean"])
    b_len = tokenizeLenSeries(df["response_b_clean"])
    pct_prompt = (prompt_len > max_len_prompt).mean() * 100
    pct_a = (a_len > max_len_resp).mean() * 100
    pct_b = (b_len > max_len_resp).mean() * 100
    return pd.DataFrame([{
        "split": split_name,
        "%prompt_truncated": round(pct_prompt, 2),
        "%respA_truncated": round(pct_a, 2),
        "%respB_truncated": round(pct_b, 2),
    }])

def sampleQualitative(df: pd.DataFrame, split_name: str, n: int = 50) -> pd.DataFrame:
    cols = ["prompt","prompt_clean","response_a","response_a_clean","response_b","response_b_clean"]
    avail = [c for c in cols if c in df.columns]
    out = df[avail].sample(min(n, len(df)), random_state=42).copy()
    out.insert(0, "split", split_name)
    return out

def writeAuditLog(cb_df: pd.DataFrame, trunc_df: pd.DataFrame, near_df: pd.DataFrame, notes: dict):
    # escribe o añade un log textual simple con datos clave del pipeline nuevo
    log_path = os.path.join(reports_dir, "00_analisis.log")

    def w(f, msg=""):
        f.write(msg + "\n")

    def pct(v): 
        try:
            return f"{float(v):.2f}%"
        except Exception:
            return str(v)

    with open(log_path, "a", encoding="utf-8") as log:
        w(log, "=== auditoria post-proceso (train/valid) ===")
        # umbrales/meta
        if notes:
            w(log, f"- umbrales dedup: hard_cos={notes.get('hard_cos','?')}  hard_jac={notes.get('hard_jac','?')}  near_cos={notes.get('near_cos','?')}")
            if "removed_hard_dups" in notes:
                w(log, f"- duplicados removidos (hard): {notes['removed_hard_dups']}")
            if "kept_near_dups" in notes:
                w(log, f"- near-dups retenidos: {notes['kept_near_dups']}")
            if "is_swapped_train_pct" in notes:
                w(log, f"- % filas swap en train: {notes['is_swapped_train_pct']}%")

        # balance por split
        for split in ["train","valid"]:
            sub = cb_df[cb_df["split"]==split].sort_values("label")
            if not sub.empty:
                parts = [f"lbl{int(r['label'])}={pct(r['percent'])}(n={int(r['count'])})" 
                         for _, r in sub.iterrows()]
                w(log, f"- balance {split}: " + " | ".join(parts))

        # truncado (acceso por nombre de columna, no itertuples)
        for _, r in trunc_df.iterrows():
            w(log, f"- truncado {r['split']}: "
                   f"prompt={pct(r['%prompt_truncated'])}  "
                   f"A={pct(r['%respA_truncated'])}  "
                   f"B={pct(r['%respB_truncated'])}")

        # near dups
        w(log, f"- near-duplicates listados: {len(near_df)} filas (umbral >=0.95 jaccard/cosine)")

        # observaciones
        w(log, "- observaciones: stopwords deshabilitadas en respuestas; puntuacion basica conservada(.,!?;:); swap solo en train; split por grupos de prompt")
        w(log, "")

# ejecución
def runOldSummaryIfNeeded():
    if cache_manager.exists(CACHE_KEY):
        print("CACHE_USED")
        return
    data_clean_path = os.path.join(clean_dir, "data_clean.csv")
    if not os.path.exists(data_clean_path):
        print("NO_CLEAN_FILE")
        return
    df = pd.read_csv(data_clean_path)
    dfs = summarizeDf(df)
    saveReports(dfs)
    cache_manager.create(CACHE_KEY, content="eda summary generated")
    print("DONE")

def runTrainValidSummary():
    if cache_manager.exists(CACHE_KEY_TV):
        print("CACHE_USED_TRAIN_VALID")
        return

    train_path = os.path.join(clean_dir, "train_strat.csv")
    valid_path = os.path.join(clean_dir, "valid_strat.csv")
    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        print("NO_TRAIN_VALID_FILES")
        return

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)

    # 1) balance
    cb_train = classBalance(train, "train")
    cb_valid = classBalance(valid, "valid")
    class_balance_detail = pd.concat([cb_train, cb_valid], ignore_index=True)
    class_balance_detail.to_csv(os.path.join(reports_dir, "class_balance_detail.csv"), index=False, encoding="utf-8-sig")

    # 2) longitudes por clase
    length_by_class_train = lengthStatsByClass(train)
    length_by_class_valid = lengthStatsByClass(valid)
    length_by_class_train.to_csv(os.path.join(reports_dir, "length_by_class_train.csv"), index=False, encoding="utf-8-sig")
    length_by_class_valid.to_csv(os.path.join(reports_dir, "length_by_class_valid.csv"), index=False, encoding="utf-8-sig")

    # 3) histogramas
    bins = list(range(0, 2049, 64))
    hist_prompt_len_train = histoLengths(train, "prompt_clean", bins, "train")
    hist_respA_len_train = histoLengths(train, "response_a_clean", bins, "train")
    hist_respB_len_train = histoLengths(train, "response_b_clean", bins, "train")
    hist_prompt_len_valid = histoLengths(valid, "prompt_clean", bins, "valid")
    hist_respA_len_valid = histoLengths(valid, "response_a_clean", bins, "valid")
    hist_respB_len_valid = histoLengths(valid, "response_b_clean", bins, "valid")

    hist_prompt_len_train.to_csv(os.path.join(reports_dir, "hist_prompt_len_train.csv"), index=False, encoding="utf-8-sig")
    hist_respA_len_train.to_csv(os.path.join(reports_dir, "hist_respA_len_train.csv"), index=False, encoding="utf-8-sig")
    hist_respB_len_train.to_csv(os.path.join(reports_dir, "hist_respB_len_train.csv"), index=False, encoding="utf-8-sig")
    hist_prompt_len_valid.to_csv(os.path.join(reports_dir, "hist_prompt_len_valid.csv"), index=False, encoding="utf-8-sig")
    hist_respA_len_valid.to_csv(os.path.join(reports_dir, "hist_respA_len_valid.csv"), index=False, encoding="utf-8-sig")
    hist_respB_len_valid.to_csv(os.path.join(reports_dir, "hist_respB_len_valid.csv"), index=False, encoding="utf-8-sig")

    # 4) similitud A vs B
    ab_similarity_train = computeSimilarity(train, "train")
    ab_similarity_valid = computeSimilarity(valid, "valid")
    ab_similarity_train.to_csv(os.path.join(reports_dir, "ab_similarity_train.csv"), index=False, encoding="utf-8-sig")
    ab_similarity_valid.to_csv(os.path.join(reports_dir, "ab_similarity_valid.csv"), index=False, encoding="utf-8-sig")

    # 5) near duplicates
    near_dup_train = nearDuplicates(ab_similarity_train, train, threshold=0.95)
    near_dup_valid = nearDuplicates(ab_similarity_valid, valid, threshold=0.95)
    near_duplicates_ab = pd.concat([near_dup_train.assign(split="train"), near_dup_valid.assign(split="valid")], ignore_index=True)
    near_duplicates_ab.to_csv(os.path.join(reports_dir, "near_duplicates_ab.csv"), index=False, encoding="utf-8-sig")

    # 6) truncado esperado
    trunc_train = truncationImpact(train, "train", max_len_prompt=512, max_len_resp=512)
    trunc_valid = truncationImpact(valid, "valid", max_len_prompt=512, max_len_resp=512)
    truncation_impact = pd.concat([trunc_train, trunc_valid], ignore_index=True)
    truncation_impact.to_csv(os.path.join(reports_dir, "truncation_impact_train_valid.csv"), index=False, encoding="utf-8-sig")

    # 7) muestra cualitativa
    sample_train = sampleQualitative(train, "train", n=60)
    sample_valid = sampleQualitative(valid, "valid", n=60)
    sample_texts = pd.concat([sample_train, sample_valid], ignore_index=True)
    sample_texts.to_csv(os.path.join(reports_dir, "sample_texts_train_valid.csv"), index=False, encoding="utf-8-sig")

    # 8) log textual con meta del pipeline nuevo
    split_meta_path = os.path.join(clean_dir, "split_meta.json")
    notes = {}
    if os.path.exists(split_meta_path):
        try:
            with open(split_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            thr = meta.get("thresholds", {})
            notes = {
                "hard_cos": thr.get("hard_cos"),
                "hard_jac": thr.get("hard_jac"),
                "near_cos": thr.get("near_cos"),
                "removed_hard_dups": meta.get("removed_hard_dups"),
                "kept_near_dups": meta.get("kept_near_dups"),
                "is_swapped_train_pct": meta.get("is_swapped_train_pct")
            }
        except Exception:
            notes = {}
    writeAuditLog(class_balance_detail, truncation_impact, near_duplicates_ab, notes)

    # cache ok
    cache_manager.create(CACHE_KEY_TV, content="train/valid summary generated")
    print("DONE_TRAIN_VALID")

def runAll():
    runOldSummaryIfNeeded()
    runTrainValidSummary()

if __name__ == "__main__":
    runAll()
