import os
from typing import Dict

import pandas as pd

from .paths import imagesResultsDir, listAvailableModels, latestRunDir
from .loaders import loadModelFrames
from .plots import (
    lineAccF1VsEpoch,
    lineEntropyVsEpoch,
    barsFinalClassF1,
    barsPredDistByModel,
    heatmapConfusionPerModel,
)

# orquestador: carga csv de la última corrida por modelo y genera gráficos comparativos
def main():
    out_dir = imagesResultsDir()
    models = listAvailableModels()

    models_epochs: Dict[str, pd.DataFrame] = {}
    models_class_report: Dict[str, pd.DataFrame] = {}
    models_conf: Dict[str, pd.DataFrame] = {}

    for model in models:
        run_dir = latestRunDir(model)
        if not run_dir:
            continue
        frames = loadModelFrames(run_dir)
        models_epochs[model] = frames.get("epochs")
        models_class_report[model] = frames.get("class_report")
        models_conf[model] = frames.get("confusion")

    # graficas comparativas
    lineAccF1VsEpoch(models_epochs, out_dir)
    lineEntropyVsEpoch(models_epochs, out_dir)
    barsFinalClassF1(models_class_report, out_dir)
    barsPredDistByModel(models_epochs, out_dir)
    heatmapConfusionPerModel(models_conf, out_dir)

    print(f"gráficas generadas en: {out_dir}")

if __name__ == "__main__":
    main()
