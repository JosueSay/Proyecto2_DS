import os
import pandas as pd

# deps locales
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "00_cache-manager"))
from cache_manager import CacheManager

# rutas base
project_dir = os.path.join(BASE_DIR, "..")
clean_dir = os.path.join(project_dir, "data", "clean")
reports_dir = os.path.join(project_dir, "reports", "eda")
os.makedirs(reports_dir, exist_ok=True)
cache_dir = os.path.join(project_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)

# cache
cache_config = os.path.join(project_dir, "00_cache-manager", "cache_config.yaml")
cache_manager = CacheManager(cache_config, cache_dir)
CACHE_KEY = "SummaryDone"

def summarizeDf(df: pd.DataFrame) -> dict:
    # genera resúmenes útiles y retorna dict con dataframes
    dfs = {}
    # tipos
    dtypes_df = pd.DataFrame(df.dtypes, columns=["dtype"]).reset_index().rename(columns={"index": "column"})
    dfs["dtypes"] = dtypes_df
    # nulos
    n = len(df)
    na_counts = df.isna().sum()
    na_percent = (na_counts / max(n, 1) * 100).round(2)
    na_df = pd.DataFrame({"column": na_counts.index, "na_count": na_counts.values, "na_percent": na_percent.values})
    dfs["na_overview"] = na_df
    # primeros/últimos
    dfs["head"] = df.head(10)
    dfs["tail"] = df.tail(10)
    # describe general
    dfs["describe"] = df.describe(include="all", datetime_is_numeric=True)
    # vacíos específicos en columnas limpias
    for col in ["prompt_clean", "response_a_clean", "response_b_clean"]:
        if col in df.columns:
            mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            dfs[f"empty_{col}"] = df.loc[mask].head(50)
    return dfs

def saveReports(dfs: dict):
    # guarda cada dataframe en reports/eda
    for name, d in dfs.items():
        out_path = os.path.join(reports_dir, f"{name}.csv")
        try:
            d.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception:
            # algunos (describe) pueden tener índice significativo
            d.to_csv(out_path, encoding="utf-8-sig")

def runSummary():
    if cache_manager.exists(CACHE_KEY):
        print("CACHE_USED")
        return

    data_clean_path = os.path.join(clean_dir, "data_clean.csv")
    if not os.path.exists(data_clean_path):
        print("NO_CLEAN_FILE")
        return

    df = pd.read_csv(data_clean_path)
    dfs = summarizeDf(df)
    saveReports(dfs)
    cache_manager.create(CACHE_KEY, content="eda summary generated")
    print("DONE")

if __name__ == "__main__":
    runSummary()