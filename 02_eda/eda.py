import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# cache manager
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "00_cache-manager"))
from cache_manager import CacheManager

# rutas base
project_dir = os.path.join(BASE_DIR, "..")
data_dir = os.path.join(project_dir, "data")
clean_dir = os.path.join(data_dir, "clean")
reports_dir = os.path.join(project_dir, "reports", "eda")
images_dir = os.path.join(project_dir, "images", "eda")
cache_dir = os.path.join(project_dir, "cache")
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# cache
cache_config = os.path.join(project_dir, "00_cache-manager", "cache_config.yaml")
cache_manager = CacheManager(cache_config, cache_dir)
CACHE_KEY = "EdaDone"

# paleta y estilo
PALETTE = {
    "dominante": "#1B3B5F",
    "secundario": "#2E5984",
    "mediacion": "#5C7EA3",
    "neutro": "#B0BEC5",
    "acento": "#F28C38",
    "confirmacion": "#3D8361",
    "advertencia": "#C14953",
    "fondo": "#F7F9FC",
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.titlecolor": PALETTE["dominante"],
    "axes.labelcolor": PALETTE["dominante"],
    "xtick.color": PALETTE["dominante"],
    "ytick.color": PALETTE["dominante"],
    "grid.color": PALETTE["neutro"],
    "grid.alpha": 0.12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": PALETTE["dominante"],
})

def saveCsv(df: pd.DataFrame, name: str):
    out = os.path.join(reports_dir, f"{name}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

def saveFigure(fig, name: str):
    out = os.path.join(images_dir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

def loadCleanData(path: str = None) -> pd.DataFrame:
    p = path or os.path.join(clean_dir, "data_clean.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no existe {p}")
    df = pd.read_csv(p)
    # si faltan longitudes, calcúlalas
    if "prompt_len" not in df.columns:
        df["prompt_len"] = df["prompt_clean"].astype(str).str.split().str.len()
    if "respA_len" not in df.columns:
        df["respA_len"] = df["response_a_clean"].astype(str).str.split().str.len()
    if "respB_len" not in df.columns:
        df["respB_len"] = df["response_b_clean"].astype(str).str.split().str.len()
    return df

def computeTables(df: pd.DataFrame) -> dict:
    tables = {}
    # resumen general
    n_rows, n_cols = df.shape
    nulls_total = int(df.isna().sum().sum())
    class_counts = df["label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count") if "label" in df.columns else pd.DataFrame()
    tables["eda_summary"] = pd.DataFrame([
        {"metric": "rows", "value": n_rows},
        {"metric": "cols", "value": n_cols},
        {"metric": "nulls_total", "value": nulls_total},
    ])
    if not class_counts.empty:
        saveCsv(class_counts, "class_balance")
    # stats de features clave
    keep = ["label","prompt_len","respA_len","respB_len","winner_model_a","winner_model_b","winner_tie"]
    num_cols = [c for c in keep if c in df.columns]
    if num_cols:
        tables["feature_stats"] = df[num_cols].describe(percentiles=[.05,.25,.5,.75,.95]).T.reset_index().rename(columns={"index":"feature"})
    else:
        tables["feature_stats"] = pd.DataFrame()
    # correlaciones
    num_all = df.select_dtypes(include=["number"])
    corr = num_all.corr(numeric_only=True) if hasattr(num_all, "corr") else pd.DataFrame()
    tables["correlations"] = corr.reset_index().rename(columns={"index":"feature"}) if not corr.empty else pd.DataFrame()
    # wins por modelo
    wins_a = df.loc[df.get("winner_model_a", pd.Series(dtype=int)) == 1, "model_a"].value_counts()
    wins_b = df.loc[df.get("winner_model_b", pd.Series(dtype=int)) == 1, "model_b"].value_counts()
    model_wins = wins_a.add(wins_b, fill_value=0).sort_values(ascending=False).rename("wins").reset_index().rename(columns={"index":"model"})
    tables["model_wins"] = model_wins
    return tables

def plotCorrelationHeatmap(df: pd.DataFrame):
    num = df.select_dtypes(include=["number"])
    if num.shape[1] == 0:
        return
    c = num.corr(numeric_only=True)
    cmap = LinearSegmentedColormap.from_list("eda_cmap", [PALETTE["neutro"], PALETTE["mediacion"], PALETTE["secundario"]])
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(c.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(c.columns)))
    ax.set_yticks(range(len(c.index)))
    ax.set_xticklabels(c.columns, rotation=45, ha="right")
    ax.set_yticklabels(c.index)
    ax.set_title("correlaciones (numéricas)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    saveFigure(fig, "correlation_heatmap")

def plotLabelDistribution(df: pd.DataFrame):
    if "label" not in df.columns:
        return
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(counts.index.astype(str), counts.values, color=[PALETTE["dominante"], PALETTE["acento"], PALETTE["confirmacion"]])
    ax.set_title("distribución de clases (label)")
    ax.set_xlabel("label")
    ax.set_ylabel("conteo")
    saveFigure(fig, "distribution_label")

def plotLengthHists(df: pd.DataFrame):
    for col, name in [("prompt_len","prompt_len"), ("respA_len","respA_len"), ("respB_len","respB_len")]:
        if col not in df.columns: 
            continue
        s = df[col].dropna().clip(upper=np.percentile(df[col], 99))  # recorte suave
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(s.values, bins=50, color=PALETTE["secundario"], edgecolor=PALETTE["dominante"])
        ax.set_title(f"distribución {name}")
        ax.set_xlabel("tokens (aprox por palabras)")
        ax.set_ylabel("conteo")
        saveFigure(fig, f"distribution_{name}")

def plotTopModelsWins(model_wins: pd.DataFrame, k: int = 10):
    if model_wins.empty:
        return
    top = model_wins.head(k)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(top["model"], top["wins"], color=PALETTE["mediacion"])
    ax.set_title(f"top {k} modelos por victorias")
    ax.set_xlabel("modelo")
    ax.set_ylabel("victorias")
    ax.tick_params(axis="x", rotation=45)
    saveFigure(fig, "top_models_wins")

def plotResultBars(df: pd.DataFrame):
    cols = [c for c in ["winner_model_a","winner_model_b","winner_tie"] if c in df.columns]
    if not cols:
        return
    vals = [int(df[c].sum()) for c in cols]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["A gana","B gana","Empate"][:len(vals)], vals, color=[PALETTE["dominante"], PALETTE["acento"], PALETTE["confirmacion"]][:len(vals)])
    ax.set_title("distribución de resultados")
    ax.set_ylabel("conteo")
    saveFigure(fig, "results_distribution")

def runEda():
    # cache
    if cache_manager.exists(CACHE_KEY):
        print("CACHE_USED")
        return

    df = loadCleanData(os.path.join(clean_dir, "data_clean.csv"))

    tables = computeTables(df)
    for name, t in tables.items():
        if not t.empty:
            saveCsv(t, name)

    plotCorrelationHeatmap(df)
    plotLabelDistribution(df)
    plotLengthHists(df)
    plotResultBars(df)
    if "model_wins" in tables:
        plotTopModelsWins(tables["model_wins"])

    meta = {
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "data/clean/data_clean.csv",
        "reports_dir": reports_dir,
        "images_dir": images_dir,
        "tables": list(tables.keys()),
    }
    with open(os.path.join(reports_dir, "eda_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    cache_manager.create(CACHE_KEY, content="eda generated")
    print("DONE")

if __name__ == "__main__":
    runEda()
