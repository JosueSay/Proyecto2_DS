import streamlit as st

# muestra KPIs clave
def showMetricCards(acc: float = None, macro_f1: float = None, loss: float = None, extra: dict = None):
    cols = st.columns(3)
    cols[0].metric("acc", f"{acc:.4f}" if acc is not None else "—")
    cols[1].metric("macro_f1", f"{macro_f1:.4f}" if macro_f1 is not None else "—")
    cols[2].metric("val_loss", f"{loss:.4f}" if loss is not None else "—")
    if extra:
        st.caption("otras métricas")
        ecols = st.columns(min(4, len(extra)))
        for i, (k, v) in enumerate(extra.items()):
            ecols[i % len(ecols)].metric(k, f"{v:.4f}" if isinstance(v, (int, float)) else str(v))
