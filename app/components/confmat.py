import pandas as pd
import streamlit as st
from utils.charts import confmatChart

# normaliza df de confusión a largo: ['true','pred','count']
def _normalizeConfDf(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns.str.lower())
    if {"true", "pred", "count"} <= cols:
        out = df.rename(columns={c: c.lower() for c in df.columns})
        out["true"] = out["true"].astype(str)
        out["pred"] = out["pred"].astype(str)
        out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0).astype(int)
        return out

    true_col = next((c for c in df.columns if c.lower() == "true"), None)
    if true_col is None:
        return pd.DataFrame(columns=["true", "pred", "count"])

    pred_cols = [c for c in df.columns if c != true_col and c.lower().startswith("pred_")]
    if not pred_cols:
        # si no hay prefijo, intenta usar todas menos 'true' como pred
        pred_cols = [c for c in df.columns if c != true_col]
        if not pred_cols:
            return pd.DataFrame(columns=["true", "pred", "count"])

    long_df = df.melt(id_vars=[true_col], value_vars=pred_cols, var_name="pred", value_name="count")
    long_df["pred"] = long_df["pred"].astype(str).str.replace(r"^pred_", "", regex=True)
    long_df = long_df.rename(columns={true_col: "true"})
    long_df["true"] = long_df["true"].astype(str)
    long_df["pred"] = long_df["pred"].astype(str)
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce").fillna(0).astype(int)
    return long_df[["true", "pred", "count"]]

def renderConfmat(conf_df: pd.DataFrame, title: str = "matriz de confusión"):
    if conf_df is None or conf_df.empty:
        st.info("sin datos de confusión")
        return
    norm = _normalizeConfDf(conf_df)
    if norm.empty:
        st.info("formato de confusión no reconocido")
        return
    chart = confmatChart(norm, title=title)
    st.altair_chart(chart, width="stretch")
