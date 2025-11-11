import pandas as pd
import streamlit as st

from utils.state import getAppConfig, getEnabledModels, getRunsCatalog
from utils.runs import getLatestRun, getRunByName
from utils.charts import barsChart

def readCsvSafe(path):
    # intenta leer un csv y devuelve None si algo falla
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# config básica de la página en streamlit
st.set_page_config(page_title="Compare Models", page_icon="⚖️", layout="wide")
st.title("⚖️ Compare Models")

# carga de configuración y modelos habilitados
cfg = getAppConfig()
models = getEnabledModels()
cat = getRunsCatalog()

rows = []
for m in models:
    # busca el run más reciente para el modelo
    latest = getLatestRun(m, cfg)
    if not latest:
        continue

    # obtiene los datos del run
    run = getRunByName(m, latest["name"], cfg)
    ep = readCsvSafe(run["reports_dir"] / "epochs.csv")

    # valores por defecto si no hay datos válidos
    macro_f1 = val_acc = val_loss = None

    if ep is not None and not ep.empty:
        # toma la última fila con datos no nulos en las métricas principales
        last = ep.dropna(subset=["macro_f1", "val_acc", "val_loss"], how="all").tail(1)
        # extrae cada métrica si existe y no está vacía
        macro_f1 = float(last["macro_f1"].iloc[0]) if "macro_f1" in last.columns and not last["macro_f1"].isna().all() else None
        val_acc  = float(last["val_acc"].iloc[0])  if "val_acc"  in last.columns and not last["val_acc"].isna().all()  else None
        val_loss = float(last["val_loss"].iloc[0]) if "val_loss" in last.columns and not last["val_loss"].isna().all() else None

    # guarda los resultados para cada modelo
    rows.append({"model": m, "run": latest["name"], "macro_f1": macro_f1, "val_acc": val_acc, "val_loss": val_loss})

# si no hay datos, se avisa y se corta la ejecución
if not rows:
    st.info("Sin runs para comparar")
    st.stop()

# crea un dataframe con los resultados y lo muestra
df = pd.DataFrame(rows)
st.dataframe(df, width="stretch")

# selector de métrica para comparar entre modelos
metric = st.selectbox("**Métrica**", ["macro_f1", "val_acc", "val_loss"], index=0)

# genera el gráfico de barras con la métrica seleccionada
ch = barsChart(df, x="model", y=metric)
st.altair_chart(ch, width="stretch")
