import os
import json
import pandas as pd
import sys
from core.settings import MAX_LEN_PROMPT, MAX_LEN_RESP, TOTAL_BUDGET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "00_cache-manager"))
from cache_manager import CacheManager

project_dir = os.path.join(BASE_DIR, "..")
clean_dir = os.path.join(project_dir, "data", "clean")
reports_dir = os.path.join(project_dir, "reports", "clean")
os.makedirs(reports_dir, exist_ok=True)
cache_dir = os.path.join(project_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)

cache_config = os.path.join(project_dir, "00_cache-manager", "cache_config.yaml")
cache_manager = CacheManager(cache_config, cache_dir)
CACHE_KEY = "SummaryDone"
CACHE_KEY_TV = "SummaryTrainValid"

from summary.metrics import (
    classBalance, lengthStatsByClass, histoLengths,
    computeSimilarity, nearDuplicates, truncationImpact,
    truncationImpactReal, sampleQualitative
)
from summary.reporting import summarizeDf, saveReports, writeAuditLog


def runOldSummaryIfNeeded():
    if cache_manager.exists(CACHE_KEY):
        print("CACHE_USED")
        return
    data_clean_path = os.path.join(clean_dir, "data_clean.csv")
    if not os.path.exists(data_clean_path):
        print("NO_CLEAN_FILE")
        return
    df = pd.read_csv(data_clean_path)
    dfs = summarizeDf(df)  # resumir df en varios reportes
    saveReports(dfs, reports_dir)  # guardar reportes resumidos
    cache_manager.create(CACHE_KEY, content="eda summary generated")
    print("DONE")


def runTrainValidSummary():
    if cache_manager.exists(CACHE_KEY_TV):
        print("CACHE_USED_TRAIN_VALID")
        return

    train_path = os.path.join(clean_dir, "train_strat.csv")
    valid_path = os.path.join(clean_dir, "valid_strat.csv")
    if not (os.path.exists(train_path) and os.path.exists(valid_path)):
        print("NO_TRAIN_VALID_FILES")
        return

    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)

    cb_train = classBalance(train, "train")  # balance de clases train
    cb_valid = classBalance(valid, "valid")  # balance de clases valid
    class_balance_detail = pd.concat([cb_train, cb_valid], ignore_index=True)
    class_balance_detail.to_csv(os.path.join(reports_dir, "class_balance_detail.csv"), index=False, encoding="utf-8-sig")

    # estadisticas de longitud por clase
    length_by_class_train = lengthStatsByClass(train)
    length_by_class_valid = lengthStatsByClass(valid)
    length_by_class_train.to_csv(os.path.join(reports_dir, "length_by_class_train.csv"), index=False, encoding="utf-8-sig")
    length_by_class_valid.to_csv(os.path.join(reports_dir, "length_by_class_valid.csv"), index=False, encoding="utf-8-sig")

    bins = list(range(0, 2049, 64))
    # histogramas de longitudes de prompts y respuestas
    hist_prompt_len_train = histoLengths(train, "prompt_clean", bins, "train")
    hist_respA_len_train = histoLengths(train, "response_a_clean", bins, "train")
    hist_respB_len_train = histoLengths(train, "response_b_clean", bins, "train")
    hist_prompt_len_valid = histoLengths(valid, "prompt_clean", bins, "valid")
    hist_respA_len_valid = histoLengths(valid, "response_a_clean", bins, "valid")
    hist_respB_len_valid = histoLengths(valid, "response_b_clean", bins, "valid")

    # guardar histogramas
    hist_prompt_len_train.to_csv(os.path.join(reports_dir, "hist_prompt_len_train.csv"), index=False, encoding="utf-8-sig")
    hist_respA_len_train.to_csv(os.path.join(reports_dir, "hist_respA_len_train.csv"), index=False, encoding="utf-8-sig")
    hist_respB_len_train.to_csv(os.path.join(reports_dir, "hist_respB_len_train.csv"), index=False, encoding="utf-8-sig")
    hist_prompt_len_valid.to_csv(os.path.join(reports_dir, "hist_prompt_len_valid.csv"), index=False, encoding="utf-8-sig")
    hist_respA_len_valid.to_csv(os.path.join(reports_dir, "hist_respA_len_valid.csv"), index=False, encoding="utf-8-sig")
    hist_respB_len_valid.to_csv(os.path.join(reports_dir, "hist_respB_len_valid.csv"), index=False, encoding="utf-8-sig")

    ab_similarity_train = computeSimilarity(train, "train")  # similitud entre resp A y B train
    ab_similarity_valid = computeSimilarity(valid, "valid")  # similitud entre resp A y B valid
    ab_similarity_train.to_csv(os.path.join(reports_dir, "ab_similarity_train.csv"), index=False, encoding="utf-8-sig")
    ab_similarity_valid.to_csv(os.path.join(reports_dir, "ab_similarity_valid.csv"), index=False, encoding="utf-8-sig")

    # detectar near duplicates
    near_dup_train = nearDuplicates(ab_similarity_train, train, threshold=0.95)
    near_dup_valid = nearDuplicates(ab_similarity_valid, valid, threshold=0.95)
    near_duplicates_ab = pd.concat([near_dup_train.assign(split="train"), near_dup_valid.assign(split="valid")], ignore_index=True)
    near_duplicates_ab.to_csv(os.path.join(reports_dir, "near_duplicates_ab.csv"), index=False, encoding="utf-8-sig")

    # impacto de truncado
    trunc_train = truncationImpact(train, "train", max_len_prompt=512, max_len_resp=512)
    trunc_valid = truncationImpact(valid, "valid", max_len_prompt=512, max_len_resp=512)
    truncation_impact = pd.concat([trunc_train, trunc_valid], ignore_index=True)
    truncation_impact.to_csv(os.path.join(reports_dir, "truncation_impact_train_valid.csv"), index=False, encoding="utf-8-sig")

    # truncado real con presupuesto
    trunc_train_real = truncationImpactReal(train, "train", max_len_prompt=MAX_LEN_PROMPT, max_len_resp=MAX_LEN_RESP, total_budget=TOTAL_BUDGET)
    trunc_valid_real = truncationImpactReal(valid, "valid", max_len_prompt=MAX_LEN_PROMPT, max_len_resp=MAX_LEN_RESP, total_budget=TOTAL_BUDGET)
    truncation_impact_real = pd.concat([trunc_train_real, trunc_valid_real], ignore_index=True)
    truncation_impact_real.to_csv(os.path.join(reports_dir, "truncation_impact_train_valid_real.csv"), index=False, encoding="utf-8-sig")

    # escribir log truncado real
    try:
        with open(os.path.join(reports_dir, "00_analisis.log"), "a", encoding="utf-8") as log:
            log.write("2-bis) impacto de truncado REAL (longest_first)\n")
            for _, r in truncation_impact_real.iterrows():
                log.write(
                    f"- {r['split']}: prompt={r['%prompt_truncated_real']:.2f}% | "
                    f"respA={r['%respA_truncated_real']:.2f}% | "
                    f"respB={r['%respB_truncated_real']:.2f}%\n"
                )
            log.write("\n")
    except Exception:
        pass

    # muestras cualitativas
    sample_train = sampleQualitative(train, "train", n=60)
    sample_valid = sampleQualitative(valid, "valid", n=60)
    sample_texts = pd.concat([sample_train, sample_valid], ignore_index=True)
    sample_texts.to_csv(os.path.join(reports_dir, "sample_texts_train_valid.csv"), index=False, encoding="utf-8-sig")

    split_meta_path = os.path.join(clean_dir, "split_meta.json")
    notes = {}
    try:
        meta = {}
        if os.path.exists(split_meta_path):
            with open(split_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # contar near duplicates
        nd_train = int(((ab_similarity_train["cosine_tfidf"] >= 0.95) | (ab_similarity_train["jaccard_tokens"] >= 0.95)).sum())
        nd_valid = int(((ab_similarity_valid["cosine_tfidf"] >= 0.95) | (ab_similarity_valid["jaccard_tokens"] >= 0.95)).sum())
        nd_total = int(nd_train + nd_valid)

        trunc_json = truncation_impact.to_dict(orient="records")
        trunc_real_json = truncation_impact_real.to_dict(orient="records")

        meta.update({
            "near_dups_after_split": {"train": nd_train, "valid": nd_valid, "total": nd_total},
            "truncation_after_split": trunc_json,
            "truncation_after_split_real": trunc_real_json,
        })
        with open(split_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        thr = meta.get("thresholds", {})
        notes = {
            "hard_cos": thr.get("hard_cos"),
            "hard_jac": thr.get("hard_jac"),
            "near_cos": thr.get("near_cos"),
            "removed_hard_dups": meta.get("removed_hard_dups"),
            "kept_near_dups": meta.get("kept_near_dups"),
            "is_swapped_train_pct": meta.get("is_swapped_train_pct"),
        }
    except Exception:
        notes = {}

    writeAuditLog(os.path.join(reports_dir, "00_analisis.log"), class_balance_detail, truncation_impact, near_duplicates_ab, notes)  # log final de auditoría

    cache_manager.create(CACHE_KEY_TV, content="train/valid summary generated")
    print("DONE_TRAIN_VALID")


def runAll():
    runOldSummaryIfNeeded()  # resumen general
    runTrainValidSummary()  # resumen train/valid


if __name__ == "__main__":
    runAll()
