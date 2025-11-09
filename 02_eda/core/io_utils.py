import os
import pandas as pd
import matplotlib.pyplot as plt
from core.paths import DIRS

def saveCsv(df: pd.DataFrame, name: str):
    out = os.path.join(DIRS["reports_eda_dir"], f"{name}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

def saveFigure(fig, name: str):
    out = os.path.join(DIRS["images_dir"], f"{name}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

def loadCleanData(path: str = None) -> pd.DataFrame:
    p = path or os.path.join(DIRS["clean_dir"], "data_clean.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no existe {p}")
    df = pd.read_csv(p)
    if "prompt_len" not in df.columns:
        df["prompt_len"] = df["prompt_clean"].astype(str).str.split().str.len()
    if "respA_len" not in df.columns:
        df["respA_len"] = df["response_a_clean"].astype(str).str.split().str.len()
    if "respB_len" not in df.columns:
        df["respB_len"] = df["response_b_clean"].astype(str).str.split().str.len()
    return df
