import pandas as pd
import streamlit as st

from utils.state import getAppConfig, getRunsCatalog, getEnabledModels
from utils.runs import getLatestRun, getRunByName
from utils.charts import lineChartEpochs, distChart, barsChart
from components.metric_cards import showMetricCards
from components.confmat import renderConfmat

def readCsvSafe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

st.set_page_config(page_title="Model Dashboard", page_icon="📊", layout="wide")
st.title("📊 Model Dashboard")

cfg = getAppConfig()
models = getEnabledModels()
cat = getRunsCatalog()

col_l, col_r = st.columns([1, 2])
with col_l:
    model = st.selectbox("modelo", models, index=0 if models else None)
    runs = [r["name"] for r in cat.get(model, [])] if model else []
    latest = getLatestRun(model, cfg) if model else None
    run_name = st.selectbox("run", runs, index=runs.index(latest["name"]) if (latest and latest["name"] in runs) else 0) if runs else None

with col_r:
    st.markdown("**bloques**")
    show_kpi   = st.checkbox("KPIs", True)
    show_curv  = st.checkbox("curvas (epochs)", True)
    show_conf  = st.checkbox("matriz de confusión", True)
    show_dist  = st.checkbox("distribuciones", True)
    show_cls   = st.checkbox("métricas por clase", True)

if not model or not run_name:
    st.info("elige modelo y run")
    st.stop()

run = getRunByName(model, run_name, cfg)
rdir = run["reports_dir"]

# KPIs
if show_kpi:
    ep = readCsvSafe(rdir / "epochs.csv")
    if ep is not None and not ep.empty:
        last = ep.dropna(subset=["macro_f1", "val_acc", "val_loss"], how="all").tail(1)
        acc  = float(last["val_acc"].iloc[0])  if "val_acc" in last.columns  and not last["val_acc"].isna().all()  else None
        f1m  = float(last["macro_f1"].iloc[0]) if "macro_f1" in last.columns and not last["macro_f1"].isna().all() else None
        vlos = float(last["val_loss"].iloc[0]) if "val_loss" in last.columns and not last["val_loss"].isna().all() else None
        showMetricCards(acc, f1m, vlos)
    else:
        st.info("sin epochs.csv")

# curvas
if show_curv:
    ep = readCsvSafe(rdir / "epochs.csv")
    if ep is not None and not ep.empty:
        ch = lineChartEpochs(ep, title=f"{model}/{run_name} · epochs")
        st.altair_chart(ch, width="stretch")
    else:
        st.info("sin epochs.csv")

# matriz de confusión
if show_conf:
    cf = readCsvSafe(rdir / "confusion.csv")
    if cf is not None and not cf.empty:
        renderConfmat(cf, title="confusión (valid)")
    else:
        st.info("sin confusion.csv")

# distribuciones de predicciones
if show_dist:
    pdist = readCsvSafe(rdir / "pred_distributions.csv")
    if pdist is not None and not pdist.empty:
        col = "prob" if "prob" in pdist.columns else pdist.select_dtypes("number").columns.tolist()[0]
        ch = distChart(pdist, col, title=f"pred_distributions · {col}")
        st.altair_chart(ch, width="stretch")
    else:
        st.info("sin pred_distributions.csv")

# métricas por clase
if show_cls:
    cr = readCsvSafe(rdir / "class_report.csv")
    if cr is not None and not cr.empty:
        y = "f1-score" if "f1-score" in cr.columns else cr.select_dtypes("number").columns.tolist()[0]
        x = "class" if "class" in cr.columns else cr.columns[0]
        ch = barsChart(cr, x=x, y=y, title=f"class_report · {y}")
        st.altair_chart(ch, width="stretch")
        st.dataframe(cr, width="stretch")
    else:
        st.info("sin class_report.csv")
