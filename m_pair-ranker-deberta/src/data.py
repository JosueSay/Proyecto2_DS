import re
import ast
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config_loader import loadYamlConfig, getValue

# limpia caracteres de control
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

def sanitizeText(s: str) -> str:
    s = "" if s is None else str(s)
    s = SURROGATE_RE.sub(" ", s)
    s = CONTROL_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def toStrSafe(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
        if "pandas" in globals() and pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", "ignore")
        except Exception:
            x = x.decode("latin-1", "ignore")
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            arr = ast.literal_eval(x)
            if isinstance(arr, list):
                x = "\n".join(map(str, arr))
        except Exception:
            pass
    return sanitizeText(x)

def pickColsStrict(df: pd.DataFrame, prompt_col: str, resp_a_col: str, resp_b_col: str):
    # exige columnas exactas del CSV stratificado
    need = {
        prompt_col, resp_a_col, resp_b_col, "label",
        "prompt_len", "respA_len", "respB_len"
    }
    # is_swapped solo en train
    if "is_swapped" in df.columns:
        need.add("is_swapped")
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"faltan columnas requeridas en el csv: {miss}")
    return prompt_col, resp_a_col, resp_b_col

class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.df = df.reset_index(drop=True)

        # 1) leer cfg
        pretrained_name = getValue(cfg, "pretrained_name")
        self.maxp = int(getValue(cfg, "max_len_prompt"))
        self.maxr = int(getValue(cfg, "max_len_resp"))
        use_slow = bool(getValue(cfg, "env.use_slow_tokenizer"))

        self.prompt_col = getValue(cfg, "data.prompt_col")
        self.respA_col  = getValue(cfg, "data.respA_col")
        self.respB_col  = getValue(cfg, "data.respB_col")

        # 2) tokenizador y tope del modelo
        self.tok = AutoTokenizer.from_pretrained(pretrained_name, use_fast=not use_slow)
        tok_max = getattr(self.tok, "model_max_length", 512)
        if not isinstance(tok_max, int) or tok_max <= 0:
            raise ValueError("tokenizer.model_max_length inválido")

        # 3) presupuestos desde cfg (consistentes)
        self.max_len_prompt_cfg = self.maxp
        self.max_len_resp_cfg   = self.maxr
        self.seq_max = min(self.maxp + self.maxr + 3, tok_max)

        # 4) validar columnas y flags
        self.col_p, self.col_a, self.col_b = pickColsStrict(self.df, self.prompt_col, self.respA_col, self.respB_col)
        self.has_id = "id" in self.df.columns
        self.has_swapped = "is_swapped" in self.df.columns

    def __len__(self):
        return len(self.df)

    def encodePair(self, prompt, resp):
        # truncation="longest_first" + padding="max_length" en prepare_for_model
        p = self.tok(toStrSafe(prompt), add_special_tokens=False, truncation=True, max_length=self.maxp)
        r = self.tok(toStrSafe(resp),   add_special_tokens=False, truncation=True, max_length=self.maxr)
        enc = self.tok.prepare_for_model(
            p["input_ids"], r["input_ids"],
            truncation=True,            # longest_first por defecto en pairs
            max_length=self.seq_max,    # tope duro (387)
            padding="max_length",       # no recortar luego
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

        # longitudes crudas
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
            meta["id"] = int(row["id"])

        # compat: mantiene tie_dummy (no usado)
        tie_dummy = torch.tensor(0, dtype=torch.long)
        return encA, encB, y, tie_dummy, meta

def collateFn(batch):
    # batch: lista de (encA, encB, y, tie_dummy, meta)
    encA_list, encB_list, y_list, tie_list, meta_list = zip(*batch)

    def stackDict(lst):
        keys = lst[0].keys()
        return {k: torch.stack([ex[k] for ex in lst], dim=0) for k in keys}

    encA = stackDict(encA_list)
    encB = stackDict(encB_list)
    y = torch.stack(y_list)
    tie_dummy = torch.stack(tie_list)

    # medición de presupuesto por batch (máx / media y flag de truncado)
    # usamos attention_mask para longitud efectiva
    def maskLens(enc):
        # (B, L) -> long por fila
        return enc["attention_mask"].sum(dim=1)  # int64

    lensA = maskLens(encA)
    lensB = maskLens(encB)
    # tomamos el mayor entre A/B como aproximación del par
    pair_lens = torch.maximum(lensA, lensB)
    input_ids_len_max = int(pair_lens.max().item())
    input_ids_len_mean = float(pair_lens.float().mean().item())

    # batch truncado si alguna secuencia toca el tope del collator
    seq_max_cfg = int(meta_list[0]["seq_max"])
    max_len_prompt_cfg = int(meta_list[0]["max_len_prompt_cfg"])
    max_len_resp_cfg   = int(meta_list[0]["max_len_resp_cfg"])
    truncated_batch = int((pair_lens >= seq_max_cfg).any().item())

    batch_meta = {
        "input_ids_len_max": input_ids_len_max,
        "input_ids_len_mean": input_ids_len_mean,
        "truncated_batch": truncated_batch,
        "max_len_prompt_cfg": max_len_prompt_cfg,
        "max_len_resp_cfg": max_len_resp_cfg,
        "total_budget_cfg": seq_max_cfg,
    }   
    return encA, encB, y, tie_dummy, batch_meta

# utilidades de carga
def loadCsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def loadDatasetFromYaml(yaml_path: str, split: str) -> PairDataset:
    cfg = loadYamlConfig(yaml_path)
    if split not in {"train", "valid", "test"}:
        raise ValueError("split debe ser 'train', 'valid' o 'test'")
    csv_path = getValue(cfg, f"data.{split}_csv")
    df = loadCsv(csv_path)
    return PairDataset(df, cfg)
