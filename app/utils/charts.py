import pandas as pd
import altair as alt

# líneas para epochs.csv (val_loss, val_acc, macro_f1)
def lineChartEpochs(df: pd.DataFrame, y_cols=None, title: str = ""):
    if y_cols is None:
        y_cols = [c for c in ["val_loss", "val_acc", "macro_f1"] if c in df.columns]
    if not y_cols:
        y_cols = [c for c in df.columns if c not in {"epoch", "time"}]

    base = alt.Chart(df).transform_fold(y_cols, as_=["metric", "value"])
    line = base.mark_line().encode(
        x=alt.X("epoch:Q", title="epoch"),
        y=alt.Y("value:Q", title="valor"),
        color=alt.Color("metric:N", title="métrica"),
        tooltip=["epoch:Q", "metric:N", alt.Tooltip("value:Q", format=".4f")],
    )
    rule = alt.Chart(df).mark_rule(color="#B0BEC5").encode(x="epoch:Q")
    return alt.layer(line, rule).resolve_scale(y="independent").properties(title=title, height=280)

# barras para métricas por clase (precision/recall/f1)
def barChartMetrics(metrics_df: pd.DataFrame, metric_col: str, cls_col: str = "class", title: str = ""):
    return (
        alt.Chart(metrics_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{cls_col}:N", title="clase", sort="-y"),
            y=alt.Y(f"{metric_col}:Q", title=metric_col),
            tooltip=[cls_col, alt.Tooltip(metric_col, format=".4f")],
        )
        .properties(title=title, height=240)
    )

# hist genérico
def distChart(df: pd.DataFrame, col: str, bins: int = 30, title: str = ""):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col),
            y=alt.Y("count()", title="frecuencia"),
            tooltip=[alt.Tooltip("count()", title="frecuencia")],
        )
        .properties(title=title, height=240)
    )

# matriz de confusión en largo: ['true','pred','count']
def confmatChart(df: pd.DataFrame, title: str = ""):
    df = df.copy()
    # orden estable priorizando A, B, TIE si existen
    pref = [c for c in ["A", "B", "TIE"] if c in set(df["true"]).union(set(df["pred"]))]
    others = sorted(list(set(df["true"]).union(set(df["pred"])) - set(pref)))
    order = pref + others

    base = alt.Chart(df).encode(
        x=alt.X("pred:N", title="pred", sort=order),
        y=alt.Y("true:N", title="true", sort=order),
    )

    heat = base.mark_rect().encode(
        color=alt.Color("count:Q", title="conteo", scale=alt.Scale(scheme="blues")),
        tooltip=["true:N", "pred:N", alt.Tooltip("count:Q", format=",.0f")],
    )
    txt = base.mark_text(baseline="middle").encode(text=alt.Text("count:Q", format=",.0f"))
    return (heat + txt).properties(title=title, height=320)

# barras genéricas (x cate, y num)
def barsChart(df: pd.DataFrame, x: str, y: str, title: str = ""):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", title=x, sort="-y"),
            y=alt.Y(f"{y}:Q", title=y),
            tooltip=[x, alt.Tooltip(y, format=".4f")],
        )
        .properties(title=title, height=240)
    )
    
# distribución simple (hist)
def distChart(df: pd.DataFrame, col: str, bins: int = 30, title: str = ""):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col),
            y=alt.Y("count()", title="frecuencia"),
            tooltip=[alt.Tooltip("count()", title="frecuencia")],
        )
        .properties(title=title, height=240)
    )