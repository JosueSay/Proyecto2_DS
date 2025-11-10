import pandas as pd
import streamlit as st

from utils.state import getAppConfig, getEnabledModels, getRunsCatalog
from utils.runs import getLatestRun, getRunByName
from utils.charts import barsChart

def readCsvSafe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

st.set_page_config(page_title="Compare Models", page_icon="⚖️", layout="wide")
st.title("⚖️ Compare Models")

cfg = getAppConfig()
models = getEnabledModels()
cat = getRunsCatalog()

rows = []
for m in models:
    latest = getLatestRun(m, cfg)
    if not latest:
        continue
    run = getRunByName(m, latest["name"], cfg)
    ep = readCsvSafe(run["reports_dir"] / "epochs.csv")
    macro_f1 = val_acc = val_loss = None
    if ep is not None and not ep.empty:
        last = ep.dropna(subset=["macro_f1", "val_acc", "val_loss"], how="all").tail(1)
        macro_f1 = float(last["macro_f1"].iloc[0]) if "macro_f1" in last.columns and not last["macro_f1"].isna().all() else None
        val_acc  = float(last["val_acc"].iloc[0])  if "val_acc"  in last.columns and not last["val_acc"].isna().all()  else None
        val_loss = float(last["val_loss"].iloc[0]) if "val_loss" in last.columns and not last["val_loss"].isna().all() else None
    rows.append({"model": m, "run": latest["name"], "macro_f1": macro_f1, "val_acc": val_acc, "val_loss": val_loss})

if not rows:
    st.info("sin runs para comparar")
    st.stop()

df = pd.DataFrame(rows)
st.dataframe(df, width="stretch")

metric = st.selectbox("métrica", ["macro_f1", "val_acc", "val_loss"], index=0)
ch = barsChart(df, x="model", y=metric, title=f"comparativa · {metric}")
st.altair_chart(ch, width="stretch")
