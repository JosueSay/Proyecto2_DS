import math
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from pairranker.config.loader import loadYamlConfig, getValue
from pairranker.utils.text import sanitizeText
from pairranker.utils.text import toStrSafe
from pairranker.utils.io import readCsv


def pickColsStrict(df: pd.DataFrame, prompt_col: str, resp_a_col: str, resp_b_col: str):
    # exige columnas exactas del CSV estratificado
    need = {
        prompt_col, resp_a_col, resp_b_col,
        "label", "prompt_len", "respA_len", "respB_len",
    }
    if "is_swapped" in df.columns:
        need.add("is_swapped")
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"faltan columnas requeridas en el csv: {miss}")
    return prompt_col, resp_a_col, resp_b_col


class PairDataset(Dataset):
    # dataset para par (prompt, respA) y (prompt, respB)
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.df = df.reset_index(drop=True)

        # cfg nueva (anidada)
        pretrained_name = getValue(cfg, "model.pretrained_name")
        self.maxp = int(getValue(cfg, "lengths.max_len_prompt"))
        self.maxr = int(getValue(cfg, "lengths.max_len_resp"))
        use_slow = bool(getValue(cfg, "env.use_slow_tokenizer"))

        self.prompt_col = getValue(cfg, "data.prompt_col")
        self.respA_col  = getValue(cfg, "data.respA_col")
        self.respB_col  = getValue(cfg, "data.respB_col")

        # tokenizador y tope
        self.tok = AutoTokenizer.from_pretrained(pretrained_name, use_fast=not use_slow)
        tok_max = getattr(self.tok, "model_max_length", 512)
        if not isinstance(tok_max, int) or tok_max <= 0:
            raise ValueError("tokenizer.model_max_length inválido")

        # presupuesto total del par
        self.max_len_prompt_cfg = self.maxp
        self.max_len_resp_cfg   = self.maxr
        self.seq_max = min(self.maxp + self.maxr + 3, tok_max)

        # columnas y flags
        self.col_p, self.col_a, self.col_b = pickColsStrict(self.df, self.prompt_col, self.respA_col, self.respB_col)
        self.has_id = "id" in self.df.columns
        self.has_swapped = "is_swapped" in self.df.columns

    def __len__(self):
        return len(self.df)

    def encodePair(self, prompt, resp):
        # truncado balanced: longest_first + padding fijo
        p = self.tok(toStrSafe(prompt), add_special_tokens=False, truncation=True, max_length=self.maxp)
        r = self.tok(toStrSafe(resp),   add_special_tokens=False, truncation=True, max_length=self.maxr)
        enc = self.tok.prepare_for_model(
            p["input_ids"], r["input_ids"],
            truncation=True,            # longest_first en pares
            max_length=self.seq_max,
            padding="max_length",
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, i):
        row = self.df.iloc[i]
        prompt = "" if pd.isna(row[self.col_p]) else str(row[self.col_p])
        resp_a = "" if pd.isna(row[self.col_a]) else str(row[self.col_a])
        resp_b = "" if pd.isna(row[self.col_b]) else str(row[self.col_b])

        encA = self.encodePair(prompt, resp_a)
        encB = self.encodePair(prompt, resp_b)

        # mapeo de label: 0=A, 1=B, 2=TIE
        y = torch.tensor(int(row["label"]), dtype=torch.long)

        # meta del ejemplo
        meta = {
            "prompt_len": int(row["prompt_len"]),
            "respA_len":  int(row["respA_len"]),
            "respB_len":  int(row["respB_len"]),
            "seq_max":    int(self.seq_max),
            "max_len_prompt_cfg": int(self.max_len_prompt_cfg),
            "max_len_resp_cfg":   int(self.max_len_resp_cfg),
        }
        if self.has_swapped:
            meta["is_swapped"] = int(row["is_swapped"])
        if self.has_id:
            try:
                meta["id"] = int(row["id"])
            except Exception:
                pass

        tie_dummy = torch.tensor(0, dtype=torch.long)  # compat con bucle
        return encA, encB, y, tie_dummy, meta


# utilidades de carga
def loadDatasetFromYaml(yaml_path: str, split: str) -> PairDataset:
    # crea dataset según el split leído del YAML
    cfg = loadYamlConfig(yaml_path, attach_run_dirs=False)
    if split not in {"train", "valid"}:
        raise ValueError("split debe ser 'train' o 'valid'")
    csv_path = getValue(cfg, f"data.{split}_csv")
    df = readCsv(csv_path)
    return PairDataset(df, cfg)
