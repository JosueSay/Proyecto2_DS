import pandas as pd

def computeTables(df: pd.DataFrame) -> dict:
    tables = {}

    n_rows, n_cols = df.shape
    nulls_total = int(df.isna().sum().sum())
    tables["eda_summary"] = pd.DataFrame([
        {"metric": "rows", "value": n_rows},
        {"metric": "cols", "value": n_cols},
        {"metric": "nulls_total", "value": nulls_total},
    ])

    if "label" in df.columns:
        class_counts = (
            df["label"].value_counts(dropna=False)
              .rename_axis("label")
              .reset_index(name="count")
        )
        tables["class_balance"] = class_counts

    keep = ["label","prompt_len","respA_len","respB_len","winner_model_a","winner_model_b","winner_tie"]
    num_cols = [c for c in keep if c in df.columns]
    tables["feature_stats"] = (
        df[num_cols].describe(percentiles=[.05,.25,.5,.75,.95]).T.reset_index().rename(columns={"index":"feature"})
        if num_cols else pd.DataFrame()
    )

    num_all = df.select_dtypes(include=["number"])
    corr = num_all.corr(numeric_only=True) if hasattr(num_all, "corr") else None
    tables["correlations"] = corr.reset_index().rename(columns={"index":"feature"}) if corr is not None and not corr.empty else pd.DataFrame()

    wins_a = df.loc[df.get("winner_model_a", pd.Series(dtype=int)) == 1, "model_a"].value_counts()
    wins_b = df.loc[df.get("winner_model_b", pd.Series(dtype=int)) == 1, "model_b"].value_counts()
    model_wins = wins_a.add(wins_b, fill_value=0).sort_values(ascending=False).rename("wins").reset_index().rename(columns={"index":"model"})
    tables["model_wins"] = model_wins

    return tables
