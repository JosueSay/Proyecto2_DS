import os
import pandas as pd
import streamlit as st

# utils
from utils.paths import repoRoot, repoPath
from utils.state import (
    initSession, getEnabledModels, setSelection, getRunsCatalog,
    getDevice, getAppConfig, applyGlobalTheme
)
from utils.runs import getLatestRun
from utils.cleaning import cleanBatch
from utils.charts import lineChartEpochs
from components.metric_cards import showMetricCards

# config de página
cfg = getAppConfig()
st.set_page_config(
    page_title=cfg["app"]["title"],
    page_icon=cfg["app"]["favicon"],
    layout="wide",
)
applyGlobalTheme()

# estilos (opcional)
css_path = repoPath("app", "assets", "styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# sesión
initSession()

st.title(f'🧭 {cfg["app"]["title"]}')
st.caption(f'Dispositivo: **{getDevice()}** · Tema: **{cfg["theme"]["mode"]}**')

# tabs
tab_models, tab_clean, tab_charts = st.tabs(["📈 Modelos", "🧹 Limpieza de datos", "📊 Charts (smoke)"])

# modelos
with tab_models:
    st.subheader("Estado de modelos")
    models = getEnabledModels()
    cat = getRunsCatalog()

    col_l, col_r = st.columns([1, 2])

    with col_l:
        model = st.selectbox("Modelo", models, index=0 if models else None, key="model_select")
        setSelection(model=model)

    with col_r:
        if model:
            latest = getLatestRun(model, cfg)
            latest_name = latest["name"] if latest else "—"
            st.metric("Último run", latest_name)
            st.write("Runs detectados:")

            rows = [
                {
                    "run": r["name"],
                    "results": str(r["results_dir"]),
                    "reports": str(r["reports_dir"] or "—"),
                }
                for r in cat.get(model, [])
            ]
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.info("no hay runs para este modelo.")
        else:
            st.warning("no hay modelos habilitados en `app_config.yaml`.")

# limpieza de datos
with tab_clean:
    st.subheader("Limpieza de CSVs de pares")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown("**Entrada**")
        up = st.file_uploader("sube un CSV con columnas id, prompt, response_a, response_b", type=["csv"])
        if st.button("usar data/test.csv"):
            up = open(os.path.join(repoRoot(), "data", "test.csv"), "rb")

    with col_b:
        st.markdown("**Salida (limpia)**")
        if up:
            df = pd.read_csv(up)
            df_clean = cleanBatch(df)
            st.dataframe(df_clean.head(50), width="stretch")
            st.download_button(
                "descargar limpio",
                df_clean.to_csv(index=False),
                "clean.csv",
                "text/csv",
            )
        else:
            st.info("sube un CSV o usa el botón para probar con `data/test.csv`.")

# gráficas
with tab_charts:
    cfg = getAppConfig()
    model = st.session_state.get("selected_model")
    if not model:
        st.info("elige un modelo en la pestaña principal")
    else:
        latest = getLatestRun(model, cfg)
        if not latest or not latest["reports_dir"]:
            st.warning("no hay reports para el último run")
        else:
            epochs_path = latest["reports_dir"].joinpath("epochs.csv")
            if not epochs_path.exists():
                st.warning(f"no existe {epochs_path}")
            else:
                df_epochs = pd.read_csv(epochs_path)
                last = df_epochs.dropna(subset=["val_acc", "macro_f1"], how="all").tail(1)
                acc   = float(last["val_acc"].iloc[0])  if "val_acc"  in last.columns and not last["val_acc"].isna().all()  else None
                f1m   = float(last["macro_f1"].iloc[0]) if "macro_f1" in last.columns and not last["macro_f1"].isna().all() else None
                vloss = float(last["val_loss"].iloc[0]) if "val_loss" in last.columns and not last["val_loss"].isna().all() else None
                showMetricCards(acc, f1m, vloss)

                chart = lineChartEpochs(df_epochs, title=f"{model} · epochs.csv")
                st.altair_chart(chart, width="stretch")
