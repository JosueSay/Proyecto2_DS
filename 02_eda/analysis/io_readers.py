import os
import pandas as pd
from core.paths import DIRS

def safeRead(name: str) -> pd.DataFrame:
    path = os.path.join(DIRS["reports_clean_dir"], name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"falta archivo en reports/clean: {name}")
    return pd.read_csv(path)

def safeReadClean(name: str) -> pd.DataFrame:
    path = os.path.join(DIRS["clean_dir"], name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"falta archivo: data/clean/{name}")
    return pd.read_csv(path)
