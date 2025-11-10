import streamlit as st

def getDevice():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

@st.cache_data(show_spinner=False)
def getAppConfig():
    from .paths import repoPath, loadYaml
    cfg_path = repoPath("app", "app_config.yaml")
    return loadYaml(cfg_path)

@st.cache_data(show_spinner=False)
def getRunsCatalog():
    from .runs import catalogAllRuns
    cfg = getAppConfig()
    return catalogAllRuns(cfg)

@st.cache_data(show_spinner=False)
def getEnabledModels():
    cfg = getAppConfig()
    return [k for k, v in cfg["models"].items() if v.get("enabled")]

@st.cache_resource(show_spinner=True)
def getModelResource(model: str, run_name: str):
    from .loaders import getArtifacts
    artifacts = getArtifacts(model, run_name)
    return artifacts

def initSession():
    if "selected_model" not in st.session_state:
        ms = getEnabledModels()
        st.session_state.selected_model = ms[0] if ms else None
    if "selected_run" not in st.session_state:
        st.session_state.selected_run = None

def setSelection(model: str = None, run_name: str = None):
    if model is not None:
        st.session_state.selected_model = model
    if run_name is not None:
        st.session_state.selected_run = run_name

# aplica paleta global leyendo app_config.yaml (sidebar, títulos, tablas, botones)
def applyGlobalTheme():
    cfg = getAppConfig()
    c = cfg["theme"]["colors"]
    css = f"""
    <style>
    .stApp {{ background: {c['background']}; }}
    section[data-testid="stSidebar"] > div {{
      background: #E9EEF4; color: {c['primary']};
    }}
    h1, h2, h3, h4, .stMetric label {{ color: {c['primary']}; }}
    button[role="tab"][aria-selected="true"] {{
      border-bottom: 2px solid {c['secondary']};
    }}
    .stDataFrame thead tr th {{ background: #E9EEF4; color: {c['primary']}; }}
    .stDataFrame tbody tr:nth-child(odd) {{ background: #F2F5F9; }}
    .stButton > button {{
      background: {c['primary']} !important; color: white !important; border: 0;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
