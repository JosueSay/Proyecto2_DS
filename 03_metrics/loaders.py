import os
import pandas as pd
from typing import Dict, Optional, Tuple

from .paths import expectedCsvs

def safeReadCsv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
    except Exception:
        return None

def loadModelFrames(run_dir: str) -> Dict[str, Optional[pd.DataFrame]]:
    csvs = expectedCsvs(run_dir)
    frames = {k: safeReadCsv(p) for k, p in csvs.items()}
    return frames

def pickLastEpochRow(df_epochs: pd.DataFrame) -> Optional[pd.Series]:
    if df_epochs is None or df_epochs.empty:
        return None
    if "epoch" in df_epochs.columns:
        df_epochs = df_epochs.sort_values("epoch")
    return df_epochs.tail(1).squeeze()

def extractFinalMetrics(df_epochs: pd.DataFrame) -> Dict[str, float]:
    # extrae métricas finales útiles para comparación rápida
    row = pickLastEpochRow(df_epochs)
    if row is None:
        return {}
    out = {}
    for col in ["val_acc", "macro_f1", "entropy", "dist_A", "dist_B", "dist_TIE"]:
        if col in row:
            try:
                out[col] = float(row[col])
            except Exception:
                pass
    return out
