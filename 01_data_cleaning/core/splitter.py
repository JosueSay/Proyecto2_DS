import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def makeSwappedDf(df: pd.DataFrame) -> pd.DataFrame:
    swapped = df.copy()
    # intercambio las respuestas A y B
    swapped["response_a_clean"], swapped["response_b_clean"] = df["response_b_clean"], df["response_a_clean"]
    # si existen columnas de modelo, también las intercambio
    if "model_a" in df.columns and "model_b" in df.columns:
        swapped["model_a"], swapped["model_b"] = df["model_b"], df["model_a"]
    # mapeo las etiquetas 0<->1, dejando 2 igual
    map_lbl = {0: 1, 1: 0, 2: 2}
    swapped["label"] = swapped["label"].map(map_lbl)
    df = df.copy()
    # agrego columna para identificar filas originales
    df["is_swapped"] = 0
    swapped["is_swapped"] = 1
    # concateno originales y swapped
    return pd.concat([df, swapped], axis=0, ignore_index=True)

def stratifiedGroupSplit(df: pd.DataFrame, test_size=0.1, random_state=42):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        # intento usar StratifiedGroupKFold si está disponible
        sgkf = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
        groups = df["prompt_sig"]
        X = df.index.values
        y = df["label"].values
        # tomo solo el primer split
        for train_idx, valid_idx in sgkf.split(X, y, groups):
            return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy(), {"splitter": "StratifiedGroupKFold"}
    except Exception:
        pass
    # fallback a GroupShuffleSplit si StratifiedGroupKFold falla
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["prompt_sig"]
    X = df.index.values
    for train_idx, valid_idx in gss.split(X, groups=groups):
        return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy(), {"splitter": "GroupShuffleSplit"}
