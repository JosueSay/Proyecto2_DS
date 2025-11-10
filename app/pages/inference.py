import os
import pandas as pd
import streamlit as st

from utils.paths import repoRoot
from utils.state import getAppConfig, getEnabledModels, getRunsCatalog
from utils.runs import getRunByName, getLatestRun
from utils.cleaning import cleanBatch
from utils.predict import predictBatch

st.set_page_config(page_title="Inference", page_icon="🔮", layout="wide")
st.title("🔮 Inference")

cfg = getAppConfig()
models = getEnabledModels()
cat = getRunsCatalog()

col_l, col_r = st.columns([1, 2])
with col_l:
    model = st.selectbox("modelo", models, index=0 if models else None)
    runs = [r["name"] for r in cat.get(model, [])] if model else []
    latest = getLatestRun(model, cfg) if model else None
    run_name = st.selectbox("run", runs, index=runs.index(latest["name"]) if (latest and latest["name"] in runs) else 0) if runs else None

tab_form, tab_batch = st.tabs(["formulario", "batch CSV"])

# formulario: una fila
with tab_form:
    st.subheader("par A vs B")
    prompt = st.text_area("prompt", height=120)
    col_a, col_b = st.columns(2)
    with col_a:
        resp_a = st.text_area("response_a", height=200, key="ra")
    with col_b:
        resp_b = st.text_area("response_b", height=200, key="rb")

    if st.button("predecir (single)"):
        if not (model and run_name):
            st.warning("elige modelo y run")
        elif not prompt or not resp_a or not resp_b:
            st.warning("completa los campos")
        else:
            df = pd.DataFrame([{"id": 0, "prompt": prompt, "response_a": resp_a, "response_b": resp_b}])
            df_clean = cleanBatch(df)
            run = getRunByName(model, run_name, cfg)
            out = predictBatch(df_clean, model)  # reutiliza batch para 1 fila
            st.dataframe(out, width="stretch")

# batch: csv
with tab_batch:
    st.subheader("CSV de prueba")
    up = st.file_uploader("csv con columnas: id, prompt, response_a, response_b", type=["csv"])
    if st.button("usar data/test.csv"):
        up = open(os.path.join(repoRoot(), "data", "test.csv"), "rb")

    if up:
        df = pd.read_csv(up)
        st.write("entrada")
        st.dataframe(df.head(20), width="stretch")

        df_clean = cleanBatch(df)
        st.write("limpio (preview)")
        st.dataframe(df_clean.head(20), width="stretch")

        if st.button("predecir batch"):
            if not (model and run_name):
                st.warning("elige modelo y run")
            else:
                run = getRunByName(model, run_name, cfg)
                preds = predictBatch(df_clean, model)
                st.success("listo")
                st.dataframe(preds.head(50), width="stretch")
                st.download_button("descargar predicciones", preds.to_csv(index=False), "preds.csv", "text/csv")
    else:
        st.info("sube un CSV o usa el botón de ejemplo")
