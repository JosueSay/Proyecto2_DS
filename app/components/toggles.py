import streamlit as st

# switches reutilizables para gráficos
def showEpochToggles():
    col1, col2, col3 = st.columns(3)
    smooth = col1.checkbox("suavizar (rolling)", value=False)
    show_loss = col2.checkbox("ver loss", value=True)
    log_y = col3.checkbox("escala log", value=False)
    return {"smooth": smooth, "show_loss": show_loss, "log_y": log_y}

def showMetricSelect(default_metrics=None):
    if default_metrics is None:
        default_metrics = ["val_acc", "macro_f1", "val_loss"]
    selected = st.multiselect("métricas", default_metrics, default=default_metrics)
    return {"metrics": selected}
