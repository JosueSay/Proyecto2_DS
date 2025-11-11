import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.io_utils import saveFigure
from core.style import PALETTE, setMplStyle
from io_readers import safeRead

# distribuciones de similitud antes vs después
def plotBeforeAfterCosine(sim_before: pd.DataFrame, sim_after: pd.DataFrame):
    setMplStyle()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sim_before["cosine_tfidf"], bins=40, alpha=0.55, label="antes", color=PALETTE["secundario"])
    ax.hist(sim_after["cosine_tfidf"],  bins=40, alpha=0.55, label="después", color=PALETTE["acento"])
    ax.set_title("cosine tf-idf (antes vs después)")
    ax.set_xlabel("cosine"); ax.set_ylabel("conteo")
    ax.legend()
    saveFigure(fig, "before_after_cosine_hist")

def plotBeforeAfterJaccard(sim_before: pd.DataFrame, sim_after: pd.DataFrame):
    setMplStyle()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sim_before["jaccard_tokens"], bins=40, alpha=0.55, label="antes", color=PALETTE["secundario"])
    ax.hist(sim_after["jaccard_tokens"],  bins=40, alpha=0.55, label="después", color=PALETTE["acento"])
    ax.set_title("jaccard tokens (antes vs después)")
    ax.set_xlabel("jaccard"); ax.set_ylabel("conteo")
    ax.legend()
    saveFigure(fig, "before_after_jaccard_hist")

# barras de truncado global antes vs después
def plotTruncationBars(trunc_before: dict, trunc_after_overall: dict):
    setMplStyle()
    keys = ["%prompt_truncated", "%respA_truncated", "%respB_truncated"]
    before_vals = [trunc_before[k] for k in keys]
    after_vals  = [trunc_after_overall[k] for k in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, before_vals, width=w, label="antes", color=PALETTE["mediacion"])
    ax.bar(x + w/2, after_vals,  width=w, label="después", color=PALETTE["acento"])
    ax.set_xticks(x, ["prompt", "respA", "respB"])
    ax.set_ylabel("% truncado")
    ax.set_title("truncado estimado (antes vs después)")
    ax.legend()
    saveFigure(fig, "before_after_truncation_bars")

# diferencia de longitudes por label (A vs B)
def plotLengthByLabelDiff():
    setMplStyle()
    try:
        t_train = safeRead("length_by_class_train.csv")
        t_valid = safeRead("length_by_class_valid.csv")
    except Exception:
        return

    def make_plot(df, split):
        df = df.sort_values("label")
        diff = (df["respA_len_tokens_mean"] - df["respB_len_tokens_mean"]).values
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(df["label"].astype(str), diff, color=PALETTE["secundario"])
        ax.axhline(0, ls="--", lw=1, color=PALETTE["neutro"])
        ax.set_title(f"diferencia medias |A-B| por label ({split})")
        ax.set_xlabel("label"); ax.set_ylabel("tokens (A-B)")
        saveFigure(fig, f"len_diff_by_label_{split}")

    make_plot(t_train, "train")
    make_plot(t_valid, "valid")

# colas de histogramas A vs B
def plotHistTailCompare(file_a: str, file_b: str, split: str):
    setMplStyle()
    try:
        ha = safeRead(file_a); hb = safeRead(file_b)
    except Exception:
        return
    need = {"label", "bin_left", "bin_right", "count"}
    if not need.issubset(ha.columns) or not need.issubset(hb.columns):
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = sorted(set(ha["label"]).intersection(set(hb["label"])))
    tails_a, tails_b = [], []
    for lbl in labels:
        a = ha[ha["label"] == lbl]; b = hb[hb["label"] == lbl]
        tail_a = a[a["bin_right"] > 512]["count"].sum()
        tail_b = b[b["bin_right"] > 512]["count"].sum()
        tot_a = a["count"].sum() or 1; tot_b = b["count"].sum() or 1
        tails_a.append(100 * tail_a / tot_a); tails_b.append(100 * tail_b / tot_b)

    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, tails_a, width=w, label="A", color=PALETTE["dominante"])
    ax.bar(x + w/2, tails_b, width=w, label="B", color=PALETTE["acento"])
    ax.set_xticks(x, [str(l) for l in labels])
    ax.set_ylabel("% en cola (>512)")
    ax.set_title(f"colas de longitud por label ({split})")
    ax.legend()
    saveFigure(fig, f"tails_compare_{split}")

# fuga de prompts
def plotLeakageBars(leakage: dict):
    setMplStyle()
    if not leakage:
        return
    pu_train = leakage.get("prompts_unique_train")
    pu_valid = leakage.get("prompts_unique_valid")
    inter    = leakage.get("prompts_intersection_count")
    pct      = leakage.get("prompts_intersection_pct") or 0.0

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["train", "valid", "inter"], [pu_train or 0, pu_valid or 0, inter or 0],
           color=[PALETTE["secundario"], PALETTE["confirmacion"], PALETTE["acento"]])
    ax.set_title(f"fuga de prompts (inter={pct:.2f}%)")
    ax.set_ylabel("conteo")
    saveFigure(fig, "prompt_leakage_bars")
 
# dispersión: longitud promedio vs similitud (antes/después)
def plotSimilarityScatter(sim_before: pd.DataFrame, sim_after: pd.DataFrame):
    # usa solo cosine, con jitter leve    
    setMplStyle()

    def scatter(df, title, name):
        cos = df["cosine_tfidf"].values
        x = np.arange(len(cos))
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.scatter(x, cos, s=6, alpha=0.35, color=PALETTE["secundario"])
        ax.set_title(title)
        ax.set_xlabel("fila (orden arbitrario)")
        ax.set_ylabel("cosine")
        saveFigure(fig, name)

    scatter(sim_before, "dispersión cosine (antes)", "scatter_cosine_before")
    scatter(sim_after,  "dispersión cosine (después)", "scatter_cosine_after")
