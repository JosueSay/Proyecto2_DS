import re
import ast
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config_loader import loadYamlConfig, getValue

CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

def sanitizeText(s: str) -> str:
    # limpia caracteres de control y espacios redundantes
    s = "" if s is None else str(s)
    s = SURROGATE_RE.sub(" ", s)
    s = CONTROL_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def toStrSafe(x):
    # conversión robusta a str, ignorando nan, bytes o listas serializadas
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
    # exige que las columnas existan y estén declaradas en el yaml
    need = {prompt_col, resp_a_col, resp_b_col, "label"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"faltan columnas requeridas en el csv: {miss}")
    return prompt_col, resp_a_col, resp_b_col

class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.df = df.reset_index(drop=True)

        # obtiene todos los parámetros desde el yaml (no hay defaults en código)
        try:
            pretrained_name = getValue(cfg, "pretrained_name")
            max_len_prompt = getValue(cfg, "max_len_prompt")
            max_len_resp = getValue(cfg, "max_len_resp")
            use_slow_tokenizer = getValue(cfg, "env.use_slow_tokenizer")
        except KeyError as e:
            raise KeyError(f"falta en default.yaml: {e}. agregalo al yaml y al validador") from e

        try:
            prompt_col = getValue(cfg, "data.prompt_col")
            resp_a_col = getValue(cfg, "data.respA_col")
            resp_b_col = getValue(cfg, "data.respB_col")
        except KeyError as e:
            raise KeyError(
                f"falta en default.yaml: {e}. agrega data.prompt_col, data.respA_col y data.respB_col"
            ) from e

        self.tok = AutoTokenizer.from_pretrained(pretrained_name, use_fast=not bool(use_slow_tokenizer))
        self.col_p, self.col_a, self.col_b = pickColsStrict(self.df, prompt_col, resp_a_col, resp_b_col)

        self.maxp = int(max_len_prompt)
        self.maxr = int(max_len_resp)
        tok_max = getattr(self.tok, "model_max_length", 512)
        if not isinstance(tok_max, int) or tok_max <= 0:
            raise ValueError("tokenizer.model_max_length inválido")
        self.seq_max = min(self.maxp + self.maxr + 3, tok_max)

        self.has_id = "id" in self.df.columns
        self.has_swapped = "is_swapped" in self.df.columns

    def __len__(self):
        return len(self.df)

    def encodePair(self, prompt, resp):
        enc = self.tok(
            toStrSafe(prompt),
            text_pair=toStrSafe(resp),
            truncation="longest_first",  # evita el error 'only_second'
            max_length=self.seq_max,     # respeta CLS + SEP + ...
            padding="max_length",
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
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        tie_dummy = torch.tensor(0, dtype=torch.long)  # compat
        return encA, encB, y, tie_dummy

def collateFn(batch):
    # agrupa lista de (encA, encB, y, tie_dummy) en tensores por batch
    encA_list, encB_list, y_list, tie_list = zip(*batch)

    def stackDict(lst):
        keys = lst[0].keys()
        return {k: torch.stack([ex[k] for ex in lst], dim=0) for k in keys}

    encA = stackDict(encA_list)
    encB = stackDict(encB_list)
    y = torch.stack(y_list)
    tie_dummy = torch.stack(tie_list)
    return encA, encB, y, tie_dummy

def loadCsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def loadDatasetFromYaml(yaml_path: str, split: str) -> PairDataset:
    # carga dataset según rutas declaradas en el yaml
    cfg = loadYamlConfig(yaml_path)
    if split not in {"train", "valid", "test"}:
        raise ValueError("split debe ser 'train', 'valid' o 'test'")
    try:
        csv_path = getValue(cfg, f"data.{split}_csv")
    except KeyError as e:
        raise KeyError(f"falta en default.yaml: {e}. agrega data.{split}_csv") from e
    df = loadCsv(csv_path)
    return PairDataset(df, cfg)
