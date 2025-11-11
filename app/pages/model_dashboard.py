import pandas as pd
import streamlit as st
import altair as alt

from utils.state import getAppConfig, getRunsCatalog, getEnabledModels
from utils.runs import getLatestRun
from utils.charts import lineChartEpochs, distChart, barsChart
from components.metric_cards import showMetricCards
from components.confmat import renderConfmat


def readCsvSafe(path):
    # intenta leer un csv y devuelve None si falla
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# config general de la página
st.set_page_config(page_title="Model Dashboard", page_icon="📊", layout="wide")
st.title("📊 Comparativa entre Modelos")

# carga la configuración y modelos habilitados
cfg = getAppConfig()
models = getEnabledModels()

# opciones de visualización disponibles
options = [
    "KPIs",
    "Curvas (epochs)",
    "Matriz de confusión",
    "Distribuciones de predicciones",
    "Métricas por clase",
    "Val Accuracy",
    "Macro F1",
    "Val Loss",
]

selected = st.selectbox("**Selecciona el tipo de métrica o gráfico a comparar**", options)


def getLatestRunData(model):
    # obtiene la última corrida registrada del modelo
    run = getLatestRun(model, cfg)
    if not run:
        return None

    # carga los distintos csv del reporte
    rdir = run["reports_dir"]
    return {
        "model": model,
        "epochs": readCsvSafe(rdir / "epochs.csv"),
        "confusion": readCsvSafe(rdir / "confusion.csv"),
        "distributions": readCsvSafe(rdir / "pred_distributions.csv"),
        "class_report": readCsvSafe(rdir / "class_report.csv"),
    }


# obtiene los datos de cada modelo activo
runs_data = [getLatestRunData(m) for m in models]
runs_data = [r for r in runs_data if r]  # descarta los nulos

# modo KPIs: muestra métricas globales de cada modelo
if selected == "KPIs":
    st.header("KPIs de cada modelo")
    cols = st.columns(len(runs_data))
    for i, r in enumerate(runs_data):
        with cols[i]:
            ep = r["epochs"]
            if ep is not None and not ep.empty:
                # toma la última fila con valores válidos de métricas
                last = ep.dropna(subset=["macro_f1", "val_acc", "val_loss"], how="all").tail(1)
                acc  = float(last["val_acc"].iloc[0])  if "val_acc" in last.columns  and not last["val_acc"].isna().all()  else None
                f1m  = float(last["macro_f1"].iloc[0]) if "macro_f1" in last.columns and not last["macro_f1"].isna().all() else None
                vlos = float(last["val_loss"].iloc[0]) if "val_loss" in last.columns and not last["val_loss"].isna().all() else None
                st.markdown(f"### {r['model']}")
                showMetricCards(acc, f1m, vlos)
            else:
                st.info(f"{r['model']}: Sin epochs.csv")

# modo curvas de entrenamiento/validación
elif selected == "Curvas (epochs)":
    st.header("Curvas de entrenamiento/validación por modelo")
    for r in runs_data:
        ep = r["epochs"]
        if ep is not None and not ep.empty:
            ch = lineChartEpochs(ep, title=r["model"])
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info(f"{r['model']}: Sin epochs.csv")

# modo matriz de confusión
elif selected == "Matriz de confusión":
    st.header("Matriz de confusión de cada modelo")
    cols = st.columns(len(runs_data))
    for i, r in enumerate(runs_data):
        with cols[i]:
            cf = r["confusion"]
            if cf is not None and not cf.empty:
                renderConfmat(cf, title=r["model"])
            else:
                st.info(f"{r['model']}: Sin confusion.csv")

# modo distribuciones de predicciones
elif selected == "Distribuciones de predicciones":
    st.header("Distribuciones de predicciones por modelo")
    for r in runs_data:
        pdist = r["distributions"]
        if pdist is not None and not pdist.empty:
            # usa la columna numérica más relevante (prob o la primera)
            col = "prob" if "prob" in pdist.columns else pdist.select_dtypes("number").columns.tolist()[0]
            ch = distChart(pdist, col, title=f"{r['model']} · {col}")
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info(f"{r['model']}: Sin pred_distributions.csv")

# modo métricas por clase
elif selected == "Métricas por clase":
    st.header("Métricas por clase de cada modelo")
    for r in runs_data:
        cr = r["class_report"]
        if cr is not None and not cr.empty:
            # selecciona las columnas correctas para el gráfico
            y = "f1-score" if "f1-score" in cr.columns else cr.select_dtypes("number").columns.tolist()[0]
            x = "class" if "class" in cr.columns else cr.columns[0]
            ch = barsChart(cr, x=x, y=y, title=f"{r['model']} · {y}")
            st.altair_chart(ch, use_container_width=True)
            st.dataframe(cr, use_container_width=True)
        else:
            st.info(f"{r['model']}: Sin class_report.csv")

# modo comparativo de métricas numéricas (val_acc, macro_f1 o val_loss)
elif selected in ["Val Accuracy", "Macro F1", "Val Loss"]:
    metric_map = {
        "Val Accuracy": "val_acc",
        "Macro F1": "macro_f1",
        "Val Loss": "val_loss",
    }
    metric = metric_map[selected]
    dfs = []
    for r in runs_data:
        ep = r["epochs"]
        # se asegura de que la métrica exista y tenga valores
        if ep is not None and not ep.empty and metric in ep.columns:
            tmp = ep.dropna(subset=[metric])
            if tmp.empty:
                continue
            tmp = tmp[["epoch", metric]].copy()
            tmp["model"] = r["model"]
            dfs.append(tmp)

    if dfs:
        # concatena todos los modelos para graficar juntos
        all_epochs = pd.concat(dfs, ignore_index=True)
        chart = (
            alt.Chart(all_epochs)
            .mark_line(point=True)
            .encode(
                x=alt.X("epoch:Q", title="Epoch"),
                y=alt.Y(f"{metric}:Q", title=selected),
                color=alt.Color("model:N", title="Modelo"),
                tooltip=["model", "epoch", metric]
            )
            .properties(title=f"Comparativa {selected} entre modelos")
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info(f"No hay datos disponibles para {selected}.")
