import pandas as pd
import streamlit as st
from pathlib import Path

from utils.paths import repoPath
from utils.state import getAppConfig
from utils.charts import lineChartEpochs, distChart, barsChart

def listCsvs(dir_path: Path):
    if not dir_path.exists():
        return []
    return [p for p in dir_path.glob("*.csv") if p.is_file()]

def guessChart(df: pd.DataFrame):
    cols = df.columns.tolist()
    num_cols = df.select_dtypes("number").columns.tolist()
    if {"epoch", "val_loss"}.issubset(cols):
        return lineChartEpochs(df, title="epochs-like")
    if len(num_cols) == 1:
        return distChart(df, num_cols[0], title=f"dist · {num_cols[0]}")
    if len(num_cols) >= 2:
        return barsChart(df.head(200), x=num_cols[0], y=num_cols[1], title=f"bars · {num_cols[0]} vs {num_cols[1]}")
    return None

st.set_page_config(page_title="Data Explorer", page_icon="🧪", layout="wide")
st.title("🧪 Data Explorer")

cfg = getAppConfig()
clean_dir = repoPath(cfg["paths"]["reports_root"], "clean")
eda_dir   = repoPath(cfg["paths"]["reports_root"], "eda")

tab_clean, tab_eda = st.tabs(["clean/", "eda/"])

with tab_clean:
    files = listCsvs(clean_dir)
    sel = st.selectbox("archivo (clean/)", [p.name for p in files], index=0 if files else None)
    if sel:
        df = pd.read_csv(clean_dir / sel)
        st.dataframe(df.head(200), width="stretch")
        ch = guessChart(df)
        if ch:
            st.altair_chart(ch, width="stretch")
        else:
            st.info("sin gráfico sugerido")

with tab_eda:
    files = listCsvs(eda_dir)
    sel = st.selectbox("archivo (eda/)", [p.name for p in files], index=0 if files else None, key="eda_sel")
    if sel:
        df = pd.read_csv(eda_dir / sel)
        st.dataframe(df.head(200), width="stretch")
        ch = guessChart(df)
        if ch:
            st.altair_chart(ch, width="stretch")
        else:
            st.info("sin gráfico sugerido")
