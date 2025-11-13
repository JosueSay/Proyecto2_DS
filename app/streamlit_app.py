import os
import pandas as pd
import streamlit as st

from utils.paths import repoRoot, repoPath
from utils.state import (
    initSession, getEnabledModels, setSelection, getRunsCatalog,
    getDevice, getAppConfig, applyGlobalTheme
)
from utils.runs import getLatestRun
from utils.cleaning import cleanBatch
from utils.charts import lineChartEpochs
from components.metric_cards import showMetricCards

# ===============================
# CONFIGURACIÓN INICIAL DE LA APP
# ===============================
cfg = getAppConfig()
st.set_page_config(
    page_title=cfg["app"]["title"],
    page_icon=cfg["app"]["favicon"],
    layout="wide",
)
applyGlobalTheme()

# ===============================
# APLICAR ESTILOS CORPORATIVOS
# ===============================
css_path = repoPath("app", "assets", "styles.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ===============================
# ENCABEZADO FIJO CON LOGO
# ===============================
logo_path = cfg["app"]["favicon"]
header_html = f"""
<header>
    <img src="{logo_path}" alt="Logo">
    <h1>{cfg["app"]["title"]}</h1>
</header>
<div class="main-content">
"""
st.markdown(header_html, unsafe_allow_html=True)

# ===============================
# INICIALIZACIÓN DE SESIÓN
# ===============================
initSession()

# ===============================
# CONTENIDO PRINCIPAL
# ===============================
st.title(f'🧭 {cfg["app"]["title"]}')
st.caption(f'Dispositivo: **{getDevice()}**')

# Pestañas principales
tab_overview, tab_models, tab_clean, tab_charts = st.tabs(
    ["**Overview**", "**Modelos**", "**Limpieza de datos**", "**Gráficas (mejor modelo)**"]
)

# ===============================
# OVERVIEW
# ===============================
with tab_overview:
    st.subheader("Problemática: Sintaxis de archivos CSV")
    st.markdown("""
    Los modelos de lenguaje a gran escala (**LLMs**) son sistemas de inteligencia artificial entrenados para comprender y generar texto de manera similar a un humano, y se utilizan en asistentes virtuales, buscadores y herramientas de productividad. Sin embargo, aún existe el reto de que sus respuestas realmente coincidan con lo que los usuarios consideran más útiles o correctas.

    En este marco surge **Chatbot Arena**, una plataforma en línea donde los usuarios interactúan con dos chatbots anónimos (basados en diferentes LLMs) que responden a la misma instrucción o *prompt*. Después de leer ambas respuestas, el usuario selecciona la que prefiere o puede declarar un empate. Este esquema de "batalla cara a cara" permite recopilar datos directos sobre las preferencias humanas frente a distintos modelos de IA.

    Comprender y predecir estas elecciones es fundamental porque aporta información sobre cómo las personas valoran la calidad de las respuestas más allá de lo técnico. Esto resulta clave para construir sistemas conversacionales más útiles, confiables y aceptados en contextos reales, ya que la capacidad de un modelo para adaptarse a las expectativas humanas determina su éxito en aplicaciones prácticas y en la satisfacción del usuario final.
    """)
    st.divider()
    st.markdown("### Carga de datos")
    st.write("Sube un archivo CSV que contenga las columnas: `id`, `prompt`, `response_a`, `response_b`")

# ===============================
# MODELOS
# ===============================
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
            st.metric("Última corrida", latest_name)

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

# ===============================
# LIMPIEZA DE DATOS
# ===============================
with tab_clean:
    st.subheader("Limpieza de CSVs de pares")
    col_a, col_b = st.columns([1, 2])
    up = None

    with col_a:
        st.markdown("**Entrada**")
        if st.button("Usar **data/test.csv**"):
            up = open(os.path.join(repoRoot(), "data", "test.csv"), "rb")
        else:
            up = st.file_uploader("**Sube un CSV para limpiar**", type=["csv"], key="clean_upload")

    with col_b:
        st.markdown("**Salida (limpia)**")
        if up:
            try:
                df = pd.read_csv(up)
                df_clean = cleanBatch(df)
                st.dataframe(df_clean.head(50), width="stretch")
                st.download_button(
                    "descargar limpio",
                    df_clean.to_csv(index=False),
                    "clean.csv",
                    "text/csv",
                )
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")
        else:
            st.info("Sube un CSV o usa el botón para probar con `data/test.csv`.")

# ===============================
# GRÁFICAS DEL MEJOR MODELO
# ===============================
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
                acc = float(last["val_acc"].iloc[0]) if "val_acc" in last.columns and not last["val_acc"].isna().all() else None
                f1m = float(last["macro_f1"].iloc[0]) if "macro_f1" in last.columns and not last["macro_f1"].isna().all() else None
                vloss = float(last["val_loss"].iloc[0]) if "val_loss" in last.columns and not last["val_loss"].isna().all() else None

                showMetricCards(acc, f1m, vloss)

                chart = lineChartEpochs(df_epochs, title=f"{model} · epochs.csv")
                st.altair_chart(chart, use_container_width=True)

# ===============================
# FOOTER CORPORATIVO
# ===============================
st.markdown("""
<footer>
    © 2025 | Dashboard Empresarial — Proyecto de Ciencia de Datos
</footer>
</div>
""", unsafe_allow_html=True)
