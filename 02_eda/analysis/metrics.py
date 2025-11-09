import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def computeSimFromDf(df: pd.DataFrame) -> pd.DataFrame:
    a = df["response_a_clean"].fillna("").astype(str).tolist()
    b = df["response_b_clean"].fillna("").astype(str).tolist()

    def jaccard_pair(x, y):
        sx, sy = set(x.split()), set(y.split())
        if not sx and not sy: return 1.0
        inter = len(sx & sy); union = len(sx | sy)
        return inter / max(union, 1)

    jacc = [jaccard_pair(x, y) for x, y in zip(a, b)]
    vect = TfidfVectorizer(min_df=1)
    X = vect.fit_transform(a + b)
    XA, XB = X[:len(a)], X[len(a):]
    cos_diag = (XA.multiply(XB)).sum(axis=1).A1 / (
        np.sqrt((XA.multiply(XA)).sum(axis=1).A1) * np.sqrt((XB.multiply(XB)).sum(axis=1).A1) + 1e-12
    )
    return pd.DataFrame({"cosine_tfidf": cos_diag, "jaccard_tokens": jacc})

def percGe(series: pd.Series, thr: float) -> float:
    if len(series) == 0: return 0.0
    return float((series >= thr).mean() * 100.0)

def truncCols(df_before: pd.DataFrame, max_len_prompt=512, max_len_resp=512) -> dict:
    tok = lambda s: s.fillna("").astype(str).str.split().map(len)
    p = tok(df_before["prompt_clean"]) if "prompt_clean" in df_before else pd.Series([], dtype=int)
    a = tok(df_before["response_a_clean"])
    b = tok(df_before["response_b_clean"])
    return {
        "%prompt_truncated": float((p > max_len_prompt).mean() * 100) if len(p) else 0.0,
        "%respA_truncated": float((a > max_len_resp).mean() * 100) if len(a) else 0.0,
        "%respB_truncated": float((b > max_len_resp).mean() * 100) if len(b) else 0.0,
        "n_rows": int(len(df_before)),
    }

def buildBeforeAfterBoard(
    before_cos_098, before_cos_0995, before_near_dups, trunc_before,
    after_cos_098, after_cos_0995, after_near_dups, trunc_after_overall
) -> pd.DataFrame:
    rows = [
        {"metric":"%cos>=0.98", "before": before_cos_098, "after": after_cos_098},
        {"metric":"%cos>=0.995", "before": before_cos_0995, "after": after_cos_0995},
        {"metric":"near_dups_total(>=0.95 cos o jac)", "before": before_near_dups, "after": after_near_dups},
        {"metric":"%prompt_truncated", "before": trunc_before["%prompt_truncated"], "after": trunc_after_overall["%prompt_truncated"]},
        {"metric":"%respA_truncated", "before": trunc_before["%respA_truncated"], "after": trunc_after_overall["%respA_truncated"]},
        {"metric":"%respB_truncated", "before": trunc_before["%respB_truncated"], "after": trunc_after_overall["%respB_truncated"]},
    ]
    return pd.DataFrame(rows)
