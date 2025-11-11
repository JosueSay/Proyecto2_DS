import yaml
import streamlit as st
from utils.state import getAppConfig
from utils.paths import repoPath

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings")

cfg = getAppConfig()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Rutas")
    st.code(
        f"""
results_root: {cfg["paths"]["results_root"]}
reports_root: {cfg["paths"]["reports_root"]}
data_root:    {cfg["paths"]["data_root"]}
images_root:  {cfg["paths"]["images_root"]}
""".strip()
    )

with col2:
    st.subheader("Modelos habilitados")
    enabled = [k for k, v in cfg["models"].items() if v.get("enabled")]
    disabled = [k for k, v in cfg["models"].items() if not v.get("enabled")]
    st.write("✅", ", ".join(enabled) if enabled else "—")
    st.write("🚫", ", ".join(disabled) if disabled else "—")

st.divider()
st.subheader("Acciones")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Limpiar cache de datos"):
        st.cache_data.clear()
        st.success("Cache de datos limpiado")

with col_b:
    if st.button("Limpiar cache de recursos"):
        st.cache_resource.clear()
        st.success("Cache de recursos limpiado")

st.info(f"Edita `app/app_config.yaml` si deseas cambiar rutas o habilitar modelos.\n\nruta: {repoPath('app','app_config.yaml')}")
