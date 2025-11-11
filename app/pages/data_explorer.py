import pandas as pd
import streamlit as st
from pathlib import Path

from utils.paths import repoPath
from utils.state import getAppConfig
from utils.charts import lineChartEpochs, distChart, barsChart


def listCsvs(dir_path: Path):
    # si el directorio no existe, no hay nada que listar
    if not dir_path.exists():
        return []
    # devuelve solo archivos csv (no carpetas)
    return [p for p in dir_path.glob("*.csv") if p.is_file()]


def guessChart(df: pd.DataFrame):
    # obtiene columnas totales y numéricas
    cols = df.columns.tolist()
    num_cols = df.select_dtypes("number").columns.tolist()

    # si tiene columnas de entrenamiento típicas → gráfico de epochs
    if {"epoch", "val_loss"}.issubset(cols):
        return lineChartEpochs(df, title="epochs-like")

    # si solo hay una columna numérica → distribución
    if len(num_cols) == 1:
        return distChart(df, num_cols[0], title=f"dist · {num_cols[0]}")

    # si hay al menos dos → gráfico de barras comparando las primeras 200 filas
    if len(num_cols) >= 2:
        return barsChart(df.head(200), x=num_cols[0], y=num_cols[1], title=f"{num_cols[0]} vs {num_cols[1]}")

    # si no hay columnas numéricas o nada relevante → sin gráfico
    return None


# configuración básica de la página
st.set_page_config(page_title="Data Explorer", page_icon="🧪", layout="wide")
st.title("🧪 Data Explorer")

# carga de configuración de la app
cfg = getAppConfig()
clean_dir = repoPath(cfg["paths"]["reports_root"], "clean")
eda_dir   = repoPath(cfg["paths"]["reports_root"], "eda")

# pestañas para data limpia y análisis exploratorio
tab_clean, tab_eda = st.tabs(["**Data limpia (reports/clean)**", "**Análisis exploratorio (reports/eda)**"])

with tab_clean:
    files = listCsvs(clean_dir)
    # combo con archivos disponibles
    sel = st.selectbox("**Archivo**", [p.name for p in files], index=0 if files else None)
    if sel:
        df = pd.read_csv(clean_dir / sel)
        # muestra las primeras 200 filas
        st.dataframe(df.head(200), width="stretch")
        ch = guessChart(df)
        # muestra gráfico sugerido si aplica
        if ch:
            st.altair_chart(ch, width="stretch")
        else:
            st.info("Sin gráfico sugerido")

with tab_eda:
    files = listCsvs(eda_dir)
    sel = st.selectbox("**Archivo**", [p.name for p in files], index=0 if files else None, key="eda_sel")
    if sel:
        df = pd.read_csv(eda_dir / sel)
        st.dataframe(df.head(200), width="stretch")
        ch = guessChart(df)
        if ch:
            st.altair_chart(ch, width="stretch")
        else:
            st.info("Sin gráfico sugerido")
