import os
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "00_cache-manager"))
from cache_manager import CacheManager

# === Configuración rutas ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

CACHE_DIR = os.path.join(BASE_DIR, "..", "cache")
CACHE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "00_cache-manager", "cache_config.yaml")

cacheManager = CacheManager(CACHE_CONFIG, CACHE_DIR)
CACHE_KEY = "SummaryDone"

FILES_TO_SUMMARIZE = ["train_clean.csv", "validation_clean.csv"]
LOG_FILE = os.path.join(LOG_DIR, "data_summary.log")

def summarizeData(filePath: str) -> str:
    df = pd.read_csv(filePath)
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"NOMBRE ARCHIVO: {os.path.basename(filePath)}")
    lines.append(f"CANTIDAD FILAS: {df.shape[0]}")
    lines.append(f"CANTIDAD COLUMNAS: {df.shape[1]}\n")
    
    lines.append("HEAD (5 filas):")
    lines.append(str(df.head(5)))
    lines.append("\nTAIL (5 filas):")
    lines.append(str(df.tail(5)))
    lines.append("\nCOLUMNAS Y TIPOS:")
    lines.append(str(df.dtypes))
    lines.append("\nNAN / NULL / NONE POR COLUMNA:")
    lines.append(str(df.isnull().sum()))
    lines.append("\nFILAS REPETIDAS:")
    lines.append(str(df.duplicated().sum()))
    lines.append("\nSUMMARY ESTADÍSTICO:")
    lines.append(str(df.describe(include='all')))
    lines.append(f"{'='*80}\n")
    return "\n".join(lines)

def main():
    if cacheManager.exists(CACHE_KEY):
        return

    allLogs = []
    for fileName in FILES_TO_SUMMARIZE:
        filePath = os.path.join(DATA_DIR, fileName)
        if not os.path.exists(filePath):
            continue
        logText = summarizeData(filePath)
        allLogs.append(logText)

    if allLogs:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(allLogs))
        cacheManager.create(CACHE_KEY, content="Data summary generated successfully.")

if __name__ == "__main__":
    main()
