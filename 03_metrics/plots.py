import os
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .styles import setMplStyle, PALETTE, getModelColor, saveFig

def ensureDir(path: str):
    os.makedirs(path, exist_ok=True)

def lineAccF1VsEpoch(models_epochs: Dict[str, pd.DataFrame], out_dir: str):
    setMplStyle()
    ensureDir(out_dir)

    # val_acc vs epoch
    plt.figure(figsize=(9, 5))
    for model, df in models_epochs.items():
        if df is None or df.empty or "epoch" not in df or "val_acc" not in df:
            continue
        x = df["epoch"].values
        y = df["val_acc"].values
        plt.plot(x, y, label=model, color=getModelColor(model))
    plt.title("validación: accuracy por época")
    plt.xlabel("época"); plt.ylabel("val_acc")
    plt.legend()
    saveFig(os.path.join(out_dir, "results_acc_vs_epoch.png"))

    # macro_f1 vs epoch
    plt.figure(figsize=(9, 5))
    for model, df in models_epochs.items():
        if df is None or df.empty or "epoch" not in df or "macro_f1" not in df:
            continue
        x = df["epoch"].values
        y = df["macro_f1"].values
        plt.plot(x, y, label=model, color=getModelColor(model))
    plt.title("validación: macro f1 por época")
    plt.xlabel("época"); plt.ylabel("macro_f1")
    plt.legend()
    saveFig(os.path.join(out_dir, "results_f1_macro_vs_epoch.png"))

def lineEntropyVsEpoch(models_epochs: Dict[str, pd.DataFrame], out_dir: str):
    setMplStyle()
    ensureDir(out_dir)

    plt.figure(figsize=(9, 5))
    for model, df in models_epochs.items():
        if df is None or df.empty or "epoch" not in df or "entropy" not in df:
            continue
        plt.plot(df["epoch"].values, df["entropy"].values, label=model, color=getModelColor(model))
    plt.title("validación: entropía de predicción por época")
    plt.xlabel("época"); plt.ylabel("entropía")
    plt.legend()
    saveFig(os.path.join(out_dir, "results_entropy_vs_epoch.png"))

def barsFinalClassF1(models_class_report: Dict[str, pd.DataFrame], out_dir: str):
    setMplStyle()
    ensureDir(out_dir)

    # f1 por clase (A, B, TIE) en última época de cada modelo
    classes = ["A", "B", "TIE"]
    class_colors = {"A": PALETTE["confirmacion"], "B": PALETTE["secundario"], "TIE": PALETTE["acento"]}
    models, data = [], []

    for model, df in models_class_report.items():
        if df is None or df.empty or "class" not in df or "f1" not in df:
            continue
        models.append(model)
        row = []
        for c in classes:
            sub = df[df["class"] == c]
            v = float(sub["f1"].values[0]) if not sub.empty else np.nan
            row.append(v)
        data.append(row)

    if not data:
        return

    data = np.array(data)  # [models, classes]
    x = np.arange(len(models))
    w = 0.25

    plt.figure(figsize=(10, 5))
    for i, c in enumerate(classes):
        plt.bar(x + i*w - w, data[:, i], width=w, label=c, color=class_colors[c])
    plt.xticks(x, models)
    plt.title("f1 por clase en la última época")
    plt.ylabel("f1")
    plt.legend(title="clase")
    saveFig(os.path.join(out_dir, "results_class_f1_by_model.png"))

def barsPredDistByModel(models_epochs: Dict[str, pd.DataFrame], out_dir: str):
    setMplStyle()
    ensureDir(out_dir)

    models = []
    dA, dB, dT = [], [], []

    for model, df in models_epochs.items():
        if df is None or df.empty:
            continue
        if {"dist_A","dist_B","dist_TIE"}.issubset(df.columns):
            last = df.sort_values("epoch").tail(1).squeeze()
            models.append(model)
            dA.append(float(last["dist_A"]))
            dB.append(float(last["dist_B"]))
            dT.append(float(last["dist_TIE"]))

    if not models:
        return

    x = np.arange(len(models))
    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.25, dA, width=0.25, label="A",   color=PALETTE["confirmacion"])
    plt.bar(x + 0.00, dB, width=0.25, label="B",   color=PALETTE["secundario"])
    plt.bar(x + 0.25, dT, width=0.25, label="TIE", color=PALETTE["acento"])
    plt.xticks(x, models)
    plt.title("distribución de predicciones promedio (última época)")
    plt.ylabel("proporción")
    plt.legend()
    saveFig(os.path.join(out_dir, "results_pred_dist_by_model.png"))

def heatmapConfusionPerModel(models_conf: Dict[str, pd.DataFrame], out_dir: str):
    setMplStyle()
    ensureDir(out_dir)

    classes = ["A", "B", "TIE"]
    for model, df in models_conf.items():
        if df is None or df.empty:
            continue
        cols = ["pred_A", "pred_B", "pred_TIE"]
        if not all(c in df.columns for c in cols):
            continue
        mat = df[cols].values.astype(float)

        plt.figure(figsize=(4.8, 4.3))
        im = plt.imshow(mat, cmap="Blues", interpolation="nearest")
        plt.title(f"matriz de confusión • {model}")
        plt.xlabel("predicho"); plt.ylabel("verdadero")
        plt.xticks(np.arange(3), classes); plt.yticks(np.arange(3), classes)
        vmin, vmax = float(mat.min()), float(mat.max())
        rng = max(1e-9, vmax - vmin)
        norm = (mat - vmin) / rng
        thr = 0.5

        for i in range(3):
            for j in range(3):
                val = int(mat[i, j])
                txt_color = "white" if norm[i, j] >= thr else PALETTE["dominante"]
                plt.text(j, i, str(val), ha="center", va="center", color=txt_color)

        plt.colorbar(im, fraction=0.046, pad=0.04)
        saveFig(os.path.join(out_dir, f"results_confmat_{model}.png"))
