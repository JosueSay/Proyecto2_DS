import os
import json
import pandas as pd

from core.paths import DIRS
from core.cache import getCache
from core.style import setMplStyle
from core.io_utils import saveCsv

from analysis.io_readers import safeRead, safeReadClean
from analysis.metrics import (
    computeSimFromDf, percGe, truncCols, buildBeforeAfterBoard
)
from analysis.logging_utils import writeLine, pct
from analysis.plots import (
    plotBeforeAfterCosine, plotBeforeAfterJaccard, plotTruncationBars,
    plotLengthByLabelDiff, plotHistTailCompare, plotLeakageBars,
    plotSimilarityScatter
)

CACHE_KEY = "EdaAnalysisDone"

def runEdaAnalysis():
    cache = getCache()
    if cache.exists(CACHE_KEY):
        print("CACHE_USED")
        return

    setMplStyle()

    reports_dir = DIRS["reports_eda_dir"]
    clean_dir = DIRS["clean_dir"]
    log_path = os.path.join(reports_dir, "00_analisis.log")

    with open(log_path, "w", encoding="utf-8") as log:
        writeLine(log, "=== analisis punto 1: sesgo en datos ===")
        writeLine(log, f"reports_dir: {reports_dir}\n")

        # balance por split
        cb = safeRead("class_balance_detail.csv")
        need_cols = {"split","label","count","percent"}
        miss = need_cols - set(cb.columns)
        if miss:
            raise ValueError(f"class_balance_detail.csv sin columnas: {miss}")

        writeLine(log, "1) balance de clases por split")
        for split in ["train","valid"]:
            if split not in cb["split"].unique():
                writeLine(log, f"- {split}: no encontrado")
                continue
            sub = cb[cb["split"]==split].sort_values("label")
            lbl_map = {0:"A",1:"B",2:"TIE"}
            parts = [f"{lbl_map.get(int(r['label']), r['label'])}={pct(float(r['percent']))} (n={int(r['count'])})" for _, r in sub.iterrows()]
            writeLine(log, f"- {split}: " + " | ".join(parts))

            tie_row = sub[sub["label"]==2]
            tie_pct = float(tie_row["percent"].iloc[0]) if not tie_row.empty else 0.0
            a_row = sub[sub["label"]==0]; b_row = sub[sub["label"]==1]
            a_pct = float(a_row["percent"].iloc[0]) if not a_row.empty else 0.0
            b_pct = float(b_row["percent"].iloc[0]) if not b_row.empty else 0.0

            flags = []
            if tie_pct>50: flags.append("TIE>50%(posible sesgo fuerte)")
            if a_pct<25:  flags.append("A<25%(subrepresentada)")
            if b_pct<25:  flags.append("B<25%(subrepresentada)")
            if flags: writeLine(log, f"\tseñales {split}: " + " ; ".join(flags))
        writeLine(log, "")

        # truncation expected por split
        ti = safeRead("truncation_impact_train_valid.csv")
        cols_ti = {"split","%prompt_truncated","%respA_truncated","%respB_truncated"}
        if not cols_ti.issubset(ti.columns):
            raise ValueError("truncation_impact_train_valid.csv con columnas inesperadas")

        writeLine(log, "2) impacto de truncado esperado (después, por split)")
        for _, r in ti.iterrows():
            split = r["split"]
            p = float(r["%prompt_truncated"]); a = float(r["%respA_truncated"]); b = float(r["%respB_truncated"])
            writeLine(log, f"- {split}: prompt={pct(p)} | respA={pct(a)} | respB={pct(b)}")
            flags = []
            for name, val in [("prompt",p),("respA",a),("respB",b)]:
                if val>30: flags.append(f"{name}>30%(alto)")
                elif val>20: flags.append(f"{name}>20%(moderado)")
            if flags: writeLine(log, f"\tseñales {split}: " + " ; ".join(flags))
        writeLine(log, "")

        # similitud después
        ab_tr = safeRead("ab_similarity_train.csv")
        ab_va = safeRead("ab_similarity_valid.csv")
        for df_sim, split in [(ab_tr,"train"),(ab_va,"valid")]:
            if not {"jaccard_tokens","cosine_tfidf"}.issubset(df_sim.columns):
                raise ValueError(f"ab_similarity_{split}.csv sin columnas de similitud")
        writeLine(log, "3) similitud A vs B (cosine/jaccard) - después")
        for name, df_sim in [("train", ab_tr), ("valid", ab_va)]:
            jac_mean = float(df_sim["jaccard_tokens"].mean())
            cos_mean = float(df_sim["cosine_tfidf"].mean())
            hi_cos = (df_sim["cosine_tfidf"]>=0.9).mean()*100
            hi_jac = (df_sim["jaccard_tokens"]>=0.8).mean()*100
            writeLine(log, f"- {name}: mean_cos={cos_mean:.3f} | mean_jac={jac_mean:.3f} | %cos>=0.9={pct(hi_cos)} | %jac>=0.8={pct(hi_jac)}")
            if hi_cos>=15 or hi_jac>=15:
                writeLine(log, f"\tseñales {name}: ≥15% pares muy similares (favorece TIE)")
        writeLine(log, "")

        # near duplicates después
        nd = safeRead("near_duplicates_ab.csv")
        writeLine(log, "4) duplicados y casi duplicados (umbral >=0.95) - después")
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

        # longitudes por clase
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
                if diff<=5: writeLine(log, "\t\tseñal: medias A/B casi iguales (<=5 tokens)")
        writeLine(log, "5) longitudes por clase (A/B) - después")
        lengthCompare("length_by_class_train.csv","train")
        lengthCompare("length_by_class_valid.csv","valid")
        writeLine(log, "")

        # histogramas de colas A vs B
        writeLine(log, "6) histogramas de longitudes (A/B) - después")
        plotHistTailCompare("hist_respA_len_train.csv","hist_respB_len_train.csv","train")
        plotHistTailCompare("hist_respA_len_valid.csv","hist_respB_len_valid.csv","valid")
        writeLine(log, "")

        # muestra cualitativa
        st = safeRead("sample_texts_train_valid.csv")
        writeLine(log, "7) muestra cualitativa (antes/después)")
        def toks(s): return s.fillna("").astype(str).str.split().map(len)
        info = []
        for pair in [("prompt","prompt_clean"),("response_a","response_a_clean"),("response_b","response_b_clean")]:
            if pair[0] in st.columns and pair[1] in st.columns:
                raw_len = toks(st[pair[0]]); clean_len = toks(st[pair[1]])
                too_short = (clean_len<=3).sum()
                lost = ((raw_len>=10) & (clean_len<=3)).sum()
                info.append((pair[1], too_short, lost))
        for name, ts, lost in info:
            writeLine(log, f"\t{name}: {ts} ejemplos muy cortos (<=3 toks); {lost} con pérdida fuerte (raw>=10 y clean<=3)")
        writeLine(log, "")

        # === tablero antes vs después ===
        writeLine(log, "=== tablero: antes vs después ===")
        df_before = safeReadClean("data_clean_base.csv")
        sim_before = computeSimFromDf(df_before)
        ab_tr = safeRead("ab_similarity_train.csv")
        ab_va = safeRead("ab_similarity_valid.csv")
        sim_after = pd.concat([ab_tr, ab_va], ignore_index=True)

        before_cos_098 = percGe(sim_before["cosine_tfidf"], 0.98)
        before_cos_0995 = percGe(sim_before["cosine_tfidf"], 0.995)
        before_near_dups = int(((sim_before["cosine_tfidf"]>=0.95) | (sim_before["jaccard_tokens"]>=0.95)).sum())
        trunc_before = truncCols(df_before=df_before, max_len_prompt=512, max_len_resp=512)

        after_cos_098 = percGe(sim_after["cosine_tfidf"], 0.98)
        after_cos_0995 = percGe(sim_after["cosine_tfidf"], 0.995)
        after_near_dups = int(((sim_after["cosine_tfidf"]>=0.95) | (sim_after["jaccard_tokens"]>=0.95)).sum())

        ti_idx = ti.set_index("split")
        train_df = safeReadClean("train_strat.csv")
        valid_df = safeReadClean("valid_strat.csv")
        n_train, n_valid = len(train_df), len(valid_df)
        w_train = n_train / max(1, (n_train + n_valid))
        w_valid = n_valid / max(1, (n_train + n_valid))
        trunc_after_overall = {
            "%prompt_truncated": float(w_train*ti_idx.loc["train","%prompt_truncated"] + w_valid*ti_idx.loc["valid","%prompt_truncated"]),
            "%respA_truncated": float(w_train*ti_idx.loc["train","%respA_truncated"] + w_valid*ti_idx.loc["valid","%respA_truncated"]),
            "%respB_truncated": float(w_train*ti_idx.loc["train","%respB_truncated"] + w_valid*ti_idx.loc["valid","%respB_truncated"]),
            "n_rows": int(n_train + n_valid),
        }

        meta_path = os.path.join(clean_dir, "split_meta.json")
        leakage = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                leakage = {
                    "prompts_unique_train": meta.get("prompts_unique_train"),
                    "prompts_unique_valid": meta.get("prompts_unique_valid"),
                    "prompts_intersection_count": meta.get("prompts_intersection_count"),
                    "prompts_intersection_pct": meta.get("prompts_intersection_pct"),
                }
            except Exception:
                leakage = {}

        board = buildBeforeAfterBoard(
            before_cos_098, before_cos_0995, before_near_dups, trunc_before,
            after_cos_098, after_cos_0995, after_near_dups, trunc_after_overall
        )
        saveCsv(board, "before_after_board")

        writeLine(log, "- similitud (por fila A vs B):")
        writeLine(log, f"\t- %cos>=0.98  | antes={pct(before_cos_098)}  después={pct(after_cos_098)}")
        writeLine(log, f"\t- %cos>=0.995 | antes={pct(before_cos_0995)}  después={pct(after_cos_0995)}")
        writeLine(log, f"\t- near-dups (>=0.95 cos o jac): antes={before_near_dups}  después={after_near_dups}")

        writeLine(log, "- truncado estimado (global):")
        writeLine(log, f"\t- prompt | antes={pct(trunc_before['%prompt_truncated'])}  después={pct(trunc_after_overall['%prompt_truncated'])}")
        writeLine(log, f"\t- respA  | antes={pct(trunc_before['%respA_truncated'])}  después={pct(trunc_after_overall['%respA_truncated'])}")
        writeLine(log, f"\t- respB  | antes={pct(trunc_before['%respB_truncated'])}  después={pct(trunc_after_overall['%respB_truncated'])}")

        if leakage:
            writeLine(log, "- fuga de prompts (después del split):")
            writeLine(log, f"\t- prompts_unique_train={leakage.get('prompts_unique_train')}")
            writeLine(log, f"\t- prompts_unique_valid={leakage.get('prompts_unique_valid')}")
            writeLine(log, f"\t- prompts_intersection_count={leakage.get('prompts_intersection_count')}")
            writeLine(log, f"\t- prompts_intersection_pct={pct(leakage.get('prompts_intersection_pct') or 0.0)}")

        writeLine(log, f"\nlisto. log escrito en: {log_path}")

    # gráficas nuevas
    plotBeforeAfterCosine(sim_before, sim_after)
    plotBeforeAfterJaccard(sim_before, sim_after)
    plotTruncationBars(trunc_before, trunc_after_overall)
    plotLengthByLabelDiff()
    plotLeakageBars(leakage if 'leakage' in locals() else {})
    plotSimilarityScatter(sim_before, sim_after)

    cache.create(CACHE_KEY, content="eda analysis generated")
    print("DONE")

if __name__ == "__main__":
    runEdaAnalysis()
