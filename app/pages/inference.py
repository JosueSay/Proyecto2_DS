import os
import io
import pandas as pd
import streamlit as st

from utils.paths import repoRoot
from utils.state import getAppConfig, getEnabledModels, getRunsCatalog
from utils.runs import getRunByName, getLatestRun
from utils.cleaning import cleanBatch
from utils.predict import predictBatch

# config básica de la página de streamlit
st.set_page_config(page_title="Inference", page_icon="🔮", layout="wide")
st.title("🔮 Inference")

# inicializa variables de sesión si no existen
if "batch_bytes" not in st.session_state:
    st.session_state.batch_bytes = None
if "batch_df" not in st.session_state:
    st.session_state.batch_df = None
if "batch_df_clean" not in st.session_state:
    st.session_state.batch_df_clean = None

# carga de configuración general, modelos habilitados y catálogo de runs
cfg = getAppConfig()
models = getEnabledModels()
cat = getRunsCatalog()

col_l, col_r = st.columns([1, 2])
with col_l:
    # selector de modelo y run
    model = st.selectbox("**Modelo**", models, index=0 if models else None, key="inf_model")
    runs = [r["name"] for r in cat.get(model, [])] if model else []
    latest = getLatestRun(model, cfg) if model else None
    run_name = st.selectbox(
        "**Corrida**",
        runs,
        index=runs.index(latest["name"]) if (latest and latest["name"] in runs) else 0,
        key="inf_run"
    ) if runs else None

# dos modos: formulario o carga de batch CSV
tab_form, tab_batch = st.tabs(["**Predicción única**", "**Predicción en conjunto (CSV)**"])

with tab_form:
    st.subheader("Respuesta A vs B")
    prompt = st.text_area("**Prompt**", height=120, key="single_prompt")
    col_a, col_b = st.columns(2)
    with col_a:
        resp_a = st.text_area("**Response A**", height=200, key="single_ra")
    with col_b:
        resp_b = st.text_area("**Response B**", height=200, key="single_rb")

    # predicción para un solo ejemplo
    if st.button("Predecir", key="btn_single_predict"):
        if not (model and run_name):
            st.warning("elige modelo y run")
        elif not prompt or not resp_a or not resp_b:
            st.warning("completa los campos")
        else:
            # se arma el dataframe con el input del usuario
            df = pd.DataFrame([{
                "id": 0,
                "prompt": prompt,
                "response_a": resp_a,
                "response_b": resp_b
            }])
            # limpieza y predicción
            df_clean = cleanBatch(df)
            out = predictBatch(df_clean, model)
            st.dataframe(out, width="stretch")

with tab_batch:
    st.subheader("CSV de prueba")

    # uploader para subir un CSV con pares prompt/response
    up = st.file_uploader(
        "**Subir CSV**",
        type=["csv"],
        key="uploader_batch"
    )

    if up is not None:
        # guarda bytes en sesión para evitar recargar el archivo cada vez
        st.session_state.batch_bytes = up.getvalue()
        st.session_state.batch_df = None
        st.session_state.batch_df_clean = None

    # botón para usar dataset de ejemplo
    if st.button("Usar **data/test.csv**", key="btn_use_sample"):
        sample_path = os.path.join(repoRoot(), "data", "test.csv")
        with open(sample_path, "rb") as f:
            st.session_state.batch_bytes = f.read()
        st.session_state.batch_df = None
        st.session_state.batch_df_clean = None

    if st.session_state.batch_bytes:
        if st.session_state.batch_df is None:
            try:
                # lectura segura del CSV
                st.session_state.batch_df = pd.read_csv(io.BytesIO(st.session_state.batch_bytes))
            except Exception as e:
                st.error(f"Error leyendo CSV: {e}")
                st.session_state.batch_bytes = None
                st.stop()

        st.write("**Entrada**")
        st.dataframe(st.session_state.batch_df.head(20), width="stretch")

        # limpieza previa antes de pasar al modelo
        if st.session_state.batch_df_clean is None:
            st.session_state.batch_df_clean = cleanBatch(st.session_state.batch_df)

        st.write("**Limpio (preview)**")
        st.dataframe(st.session_state.batch_df_clean.head(20), width="stretch")

        # ejecuta predicción batch y permite descargar resultados
        if st.button("Predecir batch", key="btn_predict_batch"):
            if not (model and run_name):
                st.warning("Elige modelo y corrida")
            else:
                preds = predictBatch(st.session_state.batch_df_clean, model)
                st.success("listo")
                st.dataframe(preds.head(50), width="stretch")
                st.download_button(
                    "Descargar predicciones",
                    preds.to_csv(index=False),
                    "preds.csv",
                    "text/csv",
                    key="btn_download_preds"
                )
    else:
        st.info("Sube un CSV o usa el botón de ejemplo")
