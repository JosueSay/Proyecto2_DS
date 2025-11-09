import os
import pandas as pd

def summarizeDf(df: pd.DataFrame) -> dict:
    dfs = {}
    # tipos de datos por columna
    dtypes_df = pd.DataFrame(df.dtypes, columns=["dtype"]).reset_index().rename(columns={"index": "column"})
    dfs["dtypes"] = dtypes_df

    n = len(df)
    na_counts = df.isna().sum()
    na_percent = (na_counts / max(n, 1) * 100).round(2)
    # overview de nulos
    dfs["na_overview"] = pd.DataFrame({
        "column": na_counts.index,
        "na_count": na_counts.values,
        "na_percent": na_percent.values
    })

    # primeras 10 filas
    dfs["head"] = df.head(10)
    # últimas 10 filas
    dfs["tail"] = df.tail(10)

    # resumen general, intentando incluir datetime como numérico
    try:
        dfs["describe"] = df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        dfs["describe"] = df.describe(include="all")

    # resumen solo numérico
    try:
        dfs["describe_numeric"] = df.describe(numeric_only=True)
    except TypeError:
        num_cols = df.select_dtypes(include=["number"]).columns
        dfs["describe_numeric"] = df[num_cols].describe() if len(num_cols) else pd.DataFrame()

    # resumen de columnas datetime
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    if dt_cols:
        rows = []
        for c in dt_cols:
            s = df[c].dropna()
            rows.append({
                "column": c,
                "min": s.min() if not s.empty else None,
                "max": s.max() if not s.empty else None,
                "nunique": s.nunique()
            })
        dfs["describe_datetime"] = pd.DataFrame(rows)

    # revisar columnas de texto específicas para filas vacías
    for col in ["prompt_clean", "response_a_clean", "response_b_clean"]:
        if col in df.columns:
            mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
            dfs[f"empty_{col}"] = df.loc[mask].head(50)

    return dfs

def saveReports(dfs: dict, reports_dir: str, prefix: str = ""):
    os.makedirs(reports_dir, exist_ok=True)
    for name, d in dfs.items():
        out_name = f"{prefix}{name}.csv" if prefix else f"{name}.csv"
        out_path = os.path.join(reports_dir, out_name)
        # intentar guardar sin índice, si falla usar default
        try:
            d.to_csv(out_path, index=False, encoding="utf-8-sig")
        except Exception:
            d.to_csv(out_path, encoding="utf-8-sig")

def writeAuditLog(log_path: str, cb_df: pd.DataFrame, trunc_df: pd.DataFrame, near_df: pd.DataFrame, notes: dict):
    def w(f, msg=""):
        f.write(msg + "\n")

    def pct(v):
        try:
            return f"{float(v):.2f}%"
        except Exception:
            return str(v)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        w(log, "=== auditoria post-proceso (train/valid) ===")
        if notes:
            # umbrales y duplicados
            w(log, f"- umbrales dedup: hard_cos={notes.get('hard_cos','?')}  hard_jac={notes.get('hard_jac','?')}  near_cos={notes.get('near_cos','?')}")
            if "removed_hard_dups" in notes:
                w(log, f"- duplicados removidos (hard): {notes['removed_hard_dups']}")
            if "kept_near_dups" in notes:
                w(log, f"- near-dups retenidos: {notes['kept_near_dups']}")
            if "is_swapped_train_pct" in notes:
                w(log, f"- % filas swap en train: {notes['is_swapped_train_pct']}%")

        # balance por split
        for split in ["train","valid"]:
            sub = cb_df[cb_df["split"]==split].sort_values("label")
            if not sub.empty:
                parts = [f"lbl{int(r['label'])}={pct(r['percent'])}(n={int(r['count'])})"
                         for _, r in sub.iterrows()]
                w(log, f"- balance {split}: " + " | ".join(parts))

        # truncados por split
        for _, r in trunc_df.iterrows():
            w(log, f"- truncado {r['split']}: "
                   f"prompt={pct(r['%prompt_truncated'])}  "
                   f"A={pct(r['%respA_truncated'])}  "
                   f"B={pct(r['%respB_truncated'])}")

        # cantidad de near-duplicates
        w(log, f"- near-duplicates listados: {len(near_df)} filas (umbral >=0.95 jaccard/cosine)")
        w(log, "- observaciones: stopwords deshabilitadas en respuestas; puntuacion basica conservada(.,!?;:); swap solo en train; split por grupos de prompt")
        w(log, "")
