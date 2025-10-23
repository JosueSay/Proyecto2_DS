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
    n_rows = df.shape[0]
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"NOMBRE ARCHIVO: {os.path.basename(filePath)}")
    lines.append(f"CANTIDAD FILAS: {n_rows}")
    lines.append(f"CANTIDAD COLUMNAS: {df.shape[1]}\n")
    
    lines.append("HEAD (5 filas):")
    lines.append(str(df.head(5)))
    lines.append("\nTAIL (5 filas):")
    lines.append(str(df.tail(5)))
    lines.append("\nCOLUMNAS Y TIPOS:")
    lines.append(str(df.dtypes))
    
    # Conteo de NaN y porcentaje
    nan_counts = df.isnull().sum()
    nan_percent = (nan_counts / n_rows * 100).round(2)
    lines.append("\nNAN / NULL / NONE POR COLUMNA (cantidad y %):")
    for col in df.columns:
        lines.append(f"{col}: {nan_counts[col]} ({nan_percent[col]}%)")
    
    # Detectar strings vacíos en columnas clean
    for col in ["prompt_clean", "response_a_clean", "response_b_clean"]:
        empty_rows = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        if not empty_rows.empty:
            lines.append(f"\nFilas con NaN o strings vacíos en {col} (hasta 20 filas):")
            lines.append(str(empty_rows[["id", "model_a", "model_b", col]].head(20)))

    
    lines.append("\nFILAS REPETIDAS:")
    lines.append(str(df.duplicated().sum()))
    lines.append("\nSUMMARY ESTADÍSTICO:")
    lines.append(str(df.describe(include='all')))
    lines.append(f"{'='*80}\n")
    return "\n".join(lines)

def main():
    if cacheManager.exists(CACHE_KEY):
        print("CACHE_USED")
        sys.exit(0)

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
        print("DONE")
        sys.exit(0)
    else:
        print("NO_FILES")
        sys.exit(2)

if __name__ == "__main__":
    main()
