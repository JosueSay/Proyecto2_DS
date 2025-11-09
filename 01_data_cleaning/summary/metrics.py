import pandas as pd
import numpy as np

def tokenizeLenSeries(s: pd.Series) -> pd.Series:
    # convertir nan a string y contar tokens por fila
    return s.fillna("").astype(str).str.split().map(len)

def classBalance(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    counts = df["label"].value_counts(dropna=False).sort_index()
    total = counts.sum()
    # armar df con conteos y porcentaje por clase
    out = pd.DataFrame({
        "split": split_name,
        "label": counts.index.astype(int),
        "count": counts.values,
        "percent": (counts.values / max(total, 1) * 100).round(2)
    })
    return out

def lengthStatsByClass(df: pd.DataFrame) -> pd.DataFrame:
    # calcular largo en tokens por columna relevante
    tmp = pd.DataFrame({
        "label": df["label"].astype(int),
        "prompt_len_tokens": tokenizeLenSeries(df["prompt_clean"]),
        "respA_len_tokens": tokenizeLenSeries(df["response_a_clean"]),
        "respB_len_tokens": tokenizeLenSeries(df["response_b_clean"])
    })
    agg = {}
    for col in ["prompt_len_tokens", "respA_len_tokens", "respB_len_tokens"]:
        # agregar estadísticos y percentiles 90 y 99
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
    # renombrar medianas para consistencia
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
    # armar histograma por clase
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    a = df["response_a_clean"].fillna("").astype(str).tolist()
    b = df["response_b_clean"].fillna("").astype(str).tolist()

    # jaccard sobre tokens
    def jaccard_pair(x, y):
        sx, sy = set(x.split()), set(y.split())
        if not sx and not sy:
            return 1.0
        inter = len(sx & sy)
        union = len(sx | sy)
        return inter / max(union, 1)

    jacc = [jaccard_pair(x, y) for x, y in zip(a, b)]

    # calcular cosine similarity con tfidf
    vect = TfidfVectorizer(min_df=1)
    X = vect.fit_transform(a + b)
    XA = X[:len(a)]
    XB = X[len(a):]
    cos_diag = (XA.multiply(XB)).sum(axis=1).A1 / (
        np.sqrt((XA.multiply(XA)).sum(axis=1).A1) * np.sqrt((XB.multiply(XB)).sum(axis=1).A1) + 1e-12
    )

    # diferencia absoluta de largo en tokens
    abs_len_diff = (tokenizeLenSeries(df["response_a_clean"]) - tokenizeLenSeries(df["response_b_clean"])).abs()

    return pd.DataFrame({
        "jaccard_tokens": np.round(jacc, 6),
        "cosine_tfidf": np.round(cos_diag, 6),
        "abs_len_diff": abs_len_diff,
        "split": split_name
    })

def nearDuplicates(sim_df: pd.DataFrame, df_src: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    # filtrar duplicados cercanos según threshold
    mask = (sim_df["jaccard_tokens"] >= threshold) | (sim_df["cosine_tfidf"] >= threshold)
    sub = df_src.loc[mask.values].copy()
    sub = sub.reset_index().rename(columns={"index": "row_id"})
    sub["jaccard_tokens"] = sim_df.loc[mask, "jaccard_tokens"].values
    sub["cosine_tfidf"] = sim_df.loc[mask, "cosine_tfidf"].values

    # recortar textos largos para preview
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
    # calcular % de filas truncadas según límites
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

def truncationImpactReal(
    df: pd.DataFrame, split_name: str,
    max_len_prompt: int = 96, max_len_resp: int = 288, total_budget: int = 387
) -> pd.DataFrame:
    # truncado real considerando presupuesto total
    p = tokenizeLenSeries(df["prompt_clean"])
    a = tokenizeLenSeries(df["response_a_clean"])
    b = tokenizeLenSeries(df["response_b_clean"])

    p_capped = np.minimum(p, max_len_prompt)
    a_capped = np.minimum(a, max_len_resp)
    b_capped = np.minimum(b, max_len_resp)

    p_trunc = (p > max_len_prompt)
    over_a = (p_capped + a_capped) - total_budget
    over_b = (p_capped + b_capped) - total_budget

    a_trunc = (a > max_len_resp) | (over_a > 0)
    b_trunc = (b > max_len_resp) | (over_b > 0)

    return pd.DataFrame([{
        "split": split_name,
        "%prompt_truncated_real": round(100 * p_trunc.mean(), 2),
        "%respA_truncated_real":  round(100 * a_trunc.mean(), 2),
        "%respB_truncated_real":  round(100 * b_trunc.mean(), 2),
    }])

def sampleQualitative(df: pd.DataFrame, split_name: str, n: int = 50) -> pd.DataFrame:
    # tomar muestra aleatoria para revisión manual
    cols = ["prompt","prompt_clean","response_a","response_a_clean","response_b","response_b_clean"]
    avail = [c for c in cols if c in df.columns]
    out = df[avail].sample(min(n, len(df)), random_state=42).copy()
    out.insert(0, "split", split_name)
    return out
