import os
import pandas as pd
import numpy as np
import sys

# cache manager
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "00_cache-manager"))
from cache_manager import CacheManager

# rutas
project_dir = os.path.join(BASE_DIR, "..")
reports_dir = os.path.join(project_dir, "reports", "clean")
cache_dir = os.path.join(project_dir, "cache")
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

CACHE_KEY = "EdaAnalysisDone"
cache_config = os.path.join(project_dir, "00_cache-manager", "cache_config.yaml")
cache_manager = CacheManager(cache_config, cache_dir)

# si ya existe en cache, no hacemos nada
if cache_manager.exists(CACHE_KEY):
    print("CACHE_USED")
    sys.exit(0)

log_path = os.path.join(reports_dir, "00_analisis.log")

# escribe en archivo log
def writeLine(f, msg=""):
    f.write(msg + "\n")

# formato porcentaje
def pct(x):
    return f"{x:.2f}%"

# lectura segura de csv
def safeRead(name):
    path = os.path.join(reports_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"falta archivo: {name}")
    return pd.read_csv(path)

with open(log_path, "w", encoding="utf-8") as log:
    writeLine(log, "=== analisis punto 1: sesgo en datos ===")
    writeLine(log, f"reports_dir: {reports_dir}\n")

    # 1) balance de clases por split
    cb = safeRead("class_balance_detail.csv")
    need_cols = {"split","label","count","percent"}
    missing = need_cols - set(cb.columns)
    if missing:
        raise ValueError(f"class_balance_detail.csv sin columnas: {missing}")

    writeLine(log, "1) balance de clases por split")
    for split in ["train","valid"]:
        if split not in cb["split"].unique():
            writeLine(log, f"- {split}: no encontrado en class_balance_detail.csv")
            continue
        sub = cb[cb["split"]==split].sort_values("label")
        lbl_map = {0:"A",1:"B",2:"TIE"}
        parts = [f"{lbl_map.get(int(r['label']), r['label'])}={pct(float(r['percent']))} (n={int(r['count'])})" for _, r in sub.iterrows()]
        writeLine(log, f"- {split}: " + " | ".join(parts))

        tie_row = sub[sub["label"]==2]
        tie_pct = float(tie_row["percent"].iloc[0]) if not tie_row.empty else 0.0
        a_row = sub[sub["label"]==0]
        b_row = sub[sub["label"]==1]
        a_pct = float(a_row["percent"].iloc[0]) if not a_row.empty else 0.0
        b_pct = float(b_row["percent"].iloc[0]) if not b_row.empty else 0.0

        flags = []
        if tie_pct>50:
            flags.append("TIE>50%(posible sesgo fuerte)")
        if a_pct<25:
            flags.append("A<25%(subrepresentada)")
        if b_pct<25:
            flags.append("B<25%(subrepresentada)")
        if flags:
            writeLine(log, f"\tseñales {split}: " + " ; ".join(flags))
    writeLine(log, "")

    # 2) truncation impact
    ti = safeRead("truncation_impact_train_valid.csv")
    cols_ti = {"split","%prompt_truncated","%respA_truncated","%respB_truncated"}
    if not cols_ti.issubset(ti.columns):
        raise ValueError("truncation_impact_train_valid.csv con columnas inesperadas")

    writeLine(log, "2) impacto de truncado esperado")
    for _, r in ti.iterrows():
        split = r["split"]
        p = float(r["%prompt_truncated"])
        a = float(r["%respA_truncated"])
        b = float(r["%respB_truncated"])
        writeLine(log, f"- {split}: prompt={pct(p)} | respA={pct(a)} | respB={pct(b)}")
        flags = []
        for name, val in [("prompt",p),("respA",a),("respB",b)]:
            if val>30:
                flags.append(f"{name}>30%(alto)")
            elif val>20:
                flags.append(f"{name}>20%(moderado)")
        if flags:
            writeLine(log, f"\tseñales {split}: " + " ; ".join(flags))
    writeLine(log, "")

    # 3) similitud A vs B
    def simStats(path, split):
        df = safeRead(path)
        if not {"jaccard_tokens","cosine_tfidf"}.issubset(df.columns):
            raise ValueError(f"{os.path.basename(path)} sin columnas de similitud")
        jac_mean = float(df["jaccard_tokens"].mean())
        cos_mean = float(df["cosine_tfidf"].mean())
        hi_cos = (df["cosine_tfidf"]>=0.9).mean()*100
        hi_jac = (df["jaccard_tokens"]>=0.8).mean()*100
        writeLine(log, f"- {split}: mean_cos={cos_mean:.3f} | mean_jac={jac_mean:.3f} | %cos>=0.9={pct(hi_cos)} | %jac>=0.8={pct(hi_jac)}")
        if hi_cos>=15 or hi_jac>=15:
            writeLine(log, f"\tseñales {split}: ≥15% pares muy similares (favorece TIE)")

    writeLine(log, "3) similitud A vs B (cosine tf-idf / jaccard)")
    simStats("ab_similarity_train.csv","train")
    simStats("ab_similarity_valid.csv","valid")
    writeLine(log, "")

    # 4) near duplicates
    nd = safeRead("near_duplicates_ab.csv")
    writeLine(log, "4) duplicados y casi duplicados (umbral >=0.95)")
    total_nd = len(nd)
    writeLine(log, f"- total filas: {total_nd}")
    if total_nd>0:
        cols_show = [c for c in ["row_id","split","label","jaccard_tokens","cosine_tfidf"] if c in nd.columns]
        show = nd[cols_show].head(5).to_dict(orient="records")
        for i, row in enumerate(show,1):
            writeLine(log, f"\tej{i}: " + " | ".join([f"{k}={row[k]}" for k in row]))
        total_ref = sum(safeRead(f).shape[0] for f in ["ab_similarity_train.csv","ab_similarity_valid.csv"])
        if total_nd>=0.15*total_ref:
            writeLine(log, "\tseñal: muchos casi duplicados (>=15% del total)")
    writeLine(log, "")

    # 5) length by class
    def lengthCompare(path, split):
        df = safeRead(path)
        need = {"label","respA_len_tokens_mean","respA_len_tokens_std","respB_len_tokens_mean","respB_len_tokens_std"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(path)} sin columnas: {missing}")
        writeLine(log, f"- {split}: resumen de longitudes por label")
        for _, r in df.sort_values("label").iterrows():
            lbl = int(r["label"])
            a_mean, a_std = float(r["respA_len_tokens_mean"]), float(r["respA_len_tokens_std"])
            b_mean, b_std = float(r["respB_len_tokens_mean"]), float(r["respB_len_tokens_std"])
            diff = abs(a_mean - b_mean)
            writeLine(log, f"\tlabel={lbl} | A_mean={a_mean:.1f} (std {a_std:.1f}) | B_mean={b_mean:.1f} (std {b_std:.1f}) | |A-B|={diff:.1f}")
            if diff<=5:
                writeLine(log, "\t\tseñal: medias A/B casi iguales (<=5 tokens)")

    writeLine(log, "5) longitudes por clase (A vs B)")
    lengthCompare("length_by_class_train.csv","train")
    lengthCompare("length_by_class_valid.csv","valid")
    writeLine(log, "")

    # 6) histogramas de longitudes
    def histSignals(file_a, file_b, split):
        ha = safeRead(file_a)
        hb = safeRead(file_b)
        need = {"label","bin_left","bin_right","count"}
        if not need.issubset(ha.columns) or not need.issubset(hb.columns):
            raise ValueError("archivos de hist con columnas inesperadas")
        writeLine(log, f"- {split}: comparando hist A vs B por label")
        for lbl in sorted(set(ha["label"]).intersection(set(hb["label"]))):
            a = ha[ha["label"]==lbl].copy()
            b = hb[hb["label"]==lbl].copy()
            tail_a = a[a["bin_right"]>512]["count"].sum()
            tail_b = b[b["bin_right"]>512]["count"].sum()
            tot_a = a["count"].sum() or 1
            tot_b = b["count"].sum() or 1
            tail_pa = 100*tail_a/tot_a
            tail_pb = 100*tail_b/tot_b
            top_a = a.sort_values("count",ascending=False).head(1)[["bin_left","bin_right","count"]].iloc[0].to_dict()
            top_b = b.sort_values("count",ascending=False).head(1)[["bin_left","bin_right","count"]].iloc[0].to_dict()
            same_bin = (top_a["bin_left"]==top_b["bin_left"]) and (top_a["bin_right"]==top_b["bin_right"])
            writeLine(log, f"\tlabel={lbl} | tailA>{pct(tail_pa)} tailB>{pct(tail_pb)} | topA=[{int(top_a['bin_left'])},{int(top_a['bin_right'])}) vs topB=[{int(top_b['bin_left'])},{int(top_b['bin_right'])}) {'(idéntico)' if same_bin else ''}")
            if tail_pa>20 or tail_pb>20:
                writeLine(log, "\t\tseñal: colas largas (posible truncado/pérdida de señal)")
            if same_bin:
                writeLine(log, "\t\tseñal: concentración idéntica A/B")

    writeLine(log, "6) histogramas de longitudes (A/B)")
    histSignals("hist_respA_len_train.csv","hist_respB_len_train.csv","train")
    histSignals("hist_respA_len_valid.csv","hist_respB_len_valid.csv","valid")
    writeLine(log, "")

    # 7) sample texts
    st = safeRead("sample_texts_train_valid.csv")
    writeLine(log, "7) muestra cualitativa (antes/después)")

    def toks(s):
        return s.fillna("").astype(str).str.split().map(len)

    info = []
    for pair in [("prompt","prompt_clean"),("response_a","response_a_clean"),("response_b","response_b_clean")]:
        if pair[0] in st.columns and pair[1] in st.columns:
            raw_len = toks(st[pair[0]])
            clean_len = toks(st[pair[1]])
            too_short = (clean_len<=3).sum()
            lost = ((raw_len>=10) & (clean_len<=3)).sum()
            info.append((pair[1], too_short, lost))
    for name, ts, lost in info:
        writeLine(log, f"\t{name}: {ts} ejemplos muy cortos (<=3 toks); {lost} con pérdida fuerte vs raw (raw>=10 y clean<=3)")

    prob = []
    for pair in [("prompt","prompt_clean"),("response_a","response_a_clean"),("response_b","response_b_clean")]:
        if pair[0] in st.columns and pair[1] in st.columns:
            raw_len = toks(st[pair[0]])
            clean_len = toks(st[pair[1]])
            mask = (raw_len>=10) & (clean_len<=3)
            tmp = st.loc[mask,[ "split", pair[0], pair[1]]].head(3)
            if not tmp.empty:
                prob.append((pair[1], tmp))
    for name, dfp in prob:
        writeLine(log, f"\tejemplos {name} con posible sobre-limpieza:")
        for _, r in dfp.iterrows():
            raw_txt = str(r[dfp.columns[1]])
            clean_txt = str(r[dfp.columns[2]])
            if len(raw_txt)>120: raw_txt = raw_txt[:120]+"..."
            if len(clean_txt)>120: clean_txt = clean_txt[:120]+"..."
            writeLine(log, f"\t\t[{r['split']}] raw='{raw_txt}' | clean='{clean_txt}'")
    writeLine(log, "")

    # extras: describe_numeric / na_overview
    writeLine(log, "extras) sanity-check")
    try:
        dn = safeRead("describe_numeric.csv")
        writeLine(log, f"\t- describe_numeric.csv ok (cols={len(dn.columns)})")
    except Exception as e:
        writeLine(log, f"\t- describe_numeric.csv no disponible ({e})")
    try:
        nav = safeRead("na_overview.csv")
        na_total = int(nav["na_count"].sum()) if "na_count" in nav.columns else "?"
        writeLine(log, f"\t- na_overview.csv ok (na_total={na_total})")
    except Exception as e:
        writeLine(log, f"\t- na_overview.csv no disponible ({e})")

    writeLine(log, f"\nlisto. log escrito en: {log_path}")

cache_manager.create(CACHE_KEY, content="eda analysis generated")
print("DONE")
