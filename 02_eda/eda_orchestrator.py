import os
import json
import pandas as pd

from core.paths import DIRS
from core.cache import getCache
from core.style import setMplStyle
from core.io_utils import loadCleanData, saveCsv
from core.tables import computeTables
from core.plots import (
    plotCorrelationHeatmap,
    plotLabelDistribution,
    plotLengthHists,
    plotResultBars,
    plotTopModelsWins,
)

CACHE_KEY = "EdaDone"

def runEda():
    cache = getCache()
    if cache.exists(CACHE_KEY):
        print("CACHE_USED")
        return  # ya se hizo el eda, salir

    setMplStyle()  # aplicar estilo matplotlib

    # cargar datos limpios
    df = loadCleanData(os.path.join(DIRS["clean_dir"], "data_clean.csv"))

    tables = computeTables(df)
    for name, t in tables.items():
        if t is not None and not t.empty:
            saveCsv(t, name)  # guardar tablas no vacías

    # generar gráficos
    plotCorrelationHeatmap(df)
    plotLabelDistribution(df)
    plotLengthHists(df)
    plotResultBars(df)
    if "model_wins" in tables:
        plotTopModelsWins(tables["model_wins"])  # gráfico especial para modelos

    # guardar metadata del eda
    meta = {
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "data/clean/data_clean.csv",
        "reports_dir": DIRS["reports_eda_dir"],
        "images_dir": DIRS["images_dir"],
        "tables": list(tables.keys()),
    }

    with open(os.path.join(DIRS["reports_eda_dir"], "eda_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


    cache.create(CACHE_KEY, content="eda generated")  # marcar cache
    print("DONE")

if __name__ == "__main__":
    runEda()
