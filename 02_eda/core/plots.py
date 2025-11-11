import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from core.io_utils import saveFigure
from core.style import PALETTE, setMplStyle

def plotCorrelationHeatmap(df: pd.DataFrame):
    setMplStyle()
    num = df.select_dtypes(include=["number"])
    if num.shape[1] == 0:
        return
    c = num.corr(numeric_only=True)
    cmap = LinearSegmentedColormap.from_list(
        "eda_cmap", [PALETTE["neutro"], PALETTE["mediacion"], PALETTE["secundario"]]
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(c.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(c.columns))); ax.set_yticks(range(len(c.index)))
    ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticklabels(c.index)
    ax.set_title("correlaciones (numéricas)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    saveFigure(fig, "correlation_heatmap")

def plotLabelDistribution(df: pd.DataFrame):
    setMplStyle()
    if "label" not in df.columns:
        return
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values,
           color=[PALETTE["dominante"], PALETTE["acento"], PALETTE["confirmacion"]][: len(counts)])
    ax.set_title("distribución de clases (label)")
    ax.set_xlabel("label"); ax.set_ylabel("conteo")
    saveFigure(fig, "distribution_label")

def plotLengthHists(df: pd.DataFrame):
    setMplStyle()
    for col, name in [("prompt_len", "prompt_len"), ("respA_len", "respA_len"), ("respB_len", "respB_len")]:
        if col not in df.columns:
            continue
        s = df[col].dropna().clip(upper=np.percentile(df[col], 99))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s.values, bins=50, color=PALETTE["secundario"], edgecolor=PALETTE["dominante"])
        ax.set_title(f"distribución {name}")
        ax.set_xlabel("tokens (aprox por palabras)")
        ax.set_ylabel("conteo")
        saveFigure(fig, f"distribution_{name}")

def plotTopModelsWins(model_wins: pd.DataFrame, k: int = 10):
    setMplStyle()
    if model_wins is None or model_wins.empty:
        return
    top = model_wins.head(k)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(top["model"], top["wins"], color=PALETTE["mediacion"])
    ax.set_title(f"top {k} modelos por victorias")
    ax.set_xlabel("modelo"); ax.set_ylabel("victorias")
    ax.tick_params(axis="x", rotation=45)
    saveFigure(fig, "top_models_wins")

def plotResultBars(df: pd.DataFrame):
    setMplStyle()
    cols = [c for c in ["winner_model_a", "winner_model_b", "winner_tie"] if c in df.columns]
    if not cols:
        return
    vals = [int(df[c].sum()) for c in cols]
    labels = ["A gana", "B gana", "Empate"][: len(vals)]
    colors = [PALETTE["dominante"], PALETTE["acento"], PALETTE["confirmacion"]][: len(vals)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=colors)
    ax.set_title("distribución de resultados")
    ax.set_ylabel("conteo")
    saveFigure(fig, "results_distribution")
