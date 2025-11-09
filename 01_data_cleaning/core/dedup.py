import pandas as pd
from text_utils import infoLen

def dedupRows(df: pd.DataFrame, hard_cos=0.998, hard_jac=0.94, near_cos=0.98):
    a = df["response_a_clean"].fillna("")
    b = df["response_b_clean"].fillna("")
    from .similarity import cosineTfidf_pairs, jaccardTokens
    cos = cosineTfidf_pairs(a.tolist(), b.tolist())  # calcula similitud coseno
    jac = [jaccardTokens(x, y) for x, y in zip(a.tolist(), b.tolist())]  # similitud jaccard
    abs_len_diff = (a.str.split().map(len) - b.str.split().map(len)).abs()  # diferencia absoluta de longitud

    df = df.copy()
    df["cosine_tfidf"] = cos
    df["jaccard_tokens"] = jac
    df["abs_len_diff"] = abs_len_diff

    hard_dup = (df["cosine_tfidf"] >= hard_cos) & (df["jaccard_tokens"] >= hard_jac)  # duplicados muy claros

    # condiciones de duplicado cercano
    cond_same_1 = (df["cosine_tfidf"] >= 0.98) & (df["abs_len_diff"] <= 16)
    cond_same_2 = (df["jaccard_tokens"] >= 0.90) | ((df["cosine_tfidf"] >= 0.985) & (df["abs_len_diff"] <= 8))
    near_same_prompt = cond_same_1 | cond_same_2

    near_cross_prompt = (df["cosine_tfidf"] >= 0.995) & (df["abs_len_diff"] <= 8)  # duplicados entre prompts

    # casos donde la diferencia es mínima y label es 2
    tie_trivial = ((df["cosine_tfidf"] >= 0.97) & (df["jaccard_tokens"] >= 0.75)) | \
                  ((df["cosine_tfidf"] >= 0.985) & (df["abs_len_diff"] <= 5))
    tie_trivial_on_tie = tie_trivial & (df["label"] == 2)

    keep_mask = pd.Series(True, index=df.index)

    # iterar por cada prompt_sig para resolver duplicados
    for sig, idx in df.groupby("prompt_sig").groups.items():
        sub = df.loc[list(idx)]
        if len(sub) <= 1:
            continue
        mark = (near_same_prompt & (df.index.isin(sub.index)))
        cand = sub[mark.loc[sub.index]].copy()
        if len(cand) > 1:
            # priorizar por infoLen
            cand["info_len_sum"] = cand["response_a_clean"].map(infoLen) + cand["response_b_clean"].map(infoLen)
            best = cand["info_len_sum"].idxmax()
            drop_idx = cand.index.difference([best])
            keep_mask.loc[drop_idx] = False

    # remover duplicados cercanos y cruzados
    keep_mask = keep_mask & ~near_cross_prompt
    keep_mask = keep_mask & ~hard_dup

    # manejar casos de empate trivial
    if tie_trivial_on_tie.any():
        for sig, idx in df[tie_trivial_on_tie].groupby("prompt_sig").groups.items():
            rows = list(idx)
            rep = df.loc[rows].assign(info_len_sum=lambda r: r["response_a_clean"].map(infoLen) + r["response_b_clean"].map(infoLen))
            keep_one = rep["info_len_sum"].idxmax()
            rows.remove(keep_one)
            keep_mask.loc[rows] = False

    removed_hard = int(hard_dup.sum())
    removed_near = int((~keep_mask & ~hard_dup).sum())
    kept_near = int(((near_same_prompt | near_cross_prompt) & keep_mask).sum())

    df_pruned = df.loc[keep_mask].copy()
    meta = {
        "hard_thresholds": {"cosine": hard_cos, "jaccard": hard_jac},
        "near_threshold": 0.98,
        "soft_cross_prompt": {"cosine": 0.995, "abs_len_diff": 8},
        "removed_total": int(len(df) - len(df_pruned)),
        "removed_hard_dups": removed_hard,
        "removed_near_dups": removed_near,
        "kept_near_dups": kept_near,
    }
    return df_pruned, meta

def clusterByTemplate(df: pd.DataFrame, max_per_prompt: int = 8, cap_start: int = 10, sim_thr: float = 0.985):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    import re

    keep_indices = []
    clusters_kept = 0
    capped_prompts = 0

    for sig, g in df.groupby("prompt_sig"):
        n_g = len(g)
        if n_g == 1:
            keep_indices.extend(g.index.tolist())
            continue

        texts_raw = (g["response_a_clean"].fillna("") + " || " + g["response_b_clean"].fillna("")).tolist()
        texts_norm = [re.sub(r"[^a-z0-9]+", " ", t.lower()).strip() for t in texts_raw]  # normalizar texto
        has_token = any(bool(re.search(r"[a-z0-9]", t)) for t in texts_norm)

        if not has_token:
            # manejar prompts vacíos
            sub = g.copy()
            sub["info_len_sum"] = sub["response_a_clean"].map(infoLen) + sub["response_b_clean"].map(infoLen)
            reps = sub.sort_values("info_len_sum", ascending=False).index.tolist()
            if n_g > cap_start and len(reps) > max_per_prompt:
                capped_prompts += 1
                reps = reps[:max_per_prompt]
            keep_indices.extend(reps)
            clusters_kept += len(reps)
            continue

        vect = TfidfVectorizer(min_df=1)
        try:
            X = vect.fit_transform(texts_norm)
        except ValueError:
            # fallback si TF-IDF falla
            sub = g.copy()
            sub["info_len_sum"] = sub["response_a_clean"].map(infoLen) + sub["response_b_clean"].map(infoLen)
            reps = sub.sort_values("info_len_sum", ascending=False).index.tolist()
            if n_g > cap_start and len(reps) > max_per_prompt:
                capped_prompts += 1
                reps = reps[:max_per_prompt]
            keep_indices.extend(reps)
            clusters_kept += len(reps)
            continue

        # calcular matriz de similitud
        sims = (X * X.T).toarray()
        n = sims.shape[0]
        visited = [False] * n
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            comp = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                neigh = np.where(sims[u] >= sim_thr)[0]
                for v in neigh:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            clusters.append(comp)

        reps = []
        for comp in clusters:
            sub = g.iloc[comp].copy()
            sub["info_len_sum"] = sub["response_a_clean"].map(infoLen) + sub["response_b_clean"].map(infoLen)
            reps.append(sub.sort_values("info_len_sum", ascending=False).index[0])  # tomar más largo como representante

        clusters_kept += len(reps)

        if n_g > cap_start and len(reps) > max_per_prompt:
            capped_prompts += 1
            reps = reps[:max_per_prompt]

        keep_indices.extend(reps)

    kept_df = df.loc[sorted(set(keep_indices))].copy()
    meta = {"clusters_kept": clusters_kept, "capped_prompts": capped_prompts, "max_per_prompt": max_per_prompt}
    return kept_df, meta
