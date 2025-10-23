import os
import pandas as pd, numpy as np, torch, ast, math, re
from torch.utils.data import Dataset
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
USE_SLOW = os.getenv("USE_SLOW_TOKENIZER", "0") == "1"

CONTROL_RE   = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")  # U+D800–U+DFFF

def sanitizeText(s: str) -> str:
    s = SURROGATE_RE.sub(" ", s)
    s = CONTROL_RE.sub(" ", s)
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    s = s.replace(r"\/", "/")   # evita SyntaxWarning por secuencias escapadas del dataset
    s = re.sub(r"\s+", " ", s).strip()
    return s

def maybeJoin(x):
    """Si viene como string de lista ['a','b'] → 'a\\nb'."""
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        try:
            arr = ast.literal_eval(x)
            if isinstance(arr, list):
                return "\n".join(map(str, arr))
        except Exception:
            return x
    return x

def toPyStr(x):
    """Convierte cualquier cosa a str puro, manejando bytes, None, NaN, listas, etc."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
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
    x = maybeJoin(x)
    s = str(x)
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return s

class PairDataset(Dataset):
    """
    Dataset para comparar respuestas A y B con su prompt.
    Limpieza + coerción a texto; permite forzar tokenizer lento por env.
    """

    def __init__(self, df: pd.DataFrame, model_name: str, max_len_prompt=256, max_len_resp=700):
        self.df = df.reset_index(drop=True)

        # Tokenizer fast/slow conmutables por env
        self.tok_fast = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tok_slow = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.tok = self.tok_slow if USE_SLOW else self.tok_fast

        self.maxp, self.maxr = int(max_len_prompt), int(max_len_resp)
        tok_max = getattr(self.tok, "model_max_length", 512) or 512
        # +3 por [CLS]/[SEP]/separadores
        self.seq_max = min(self.maxp + self.maxr + 3, tok_max if tok_max > 0 else 512)

    def __len__(self):
        return len(self.df)

    def encodePair(self, prompt, resp):
        text = sanitizeText(toPyStr(prompt))
        pair = sanitizeText(toPyStr(resp))
        try:
            enc = self.tok.encode_plus(
                text=text,
                text_pair=pair,
                truncation=True,
                max_length=self.seq_max,
                padding="max_length",
                return_tensors="pt",
            )
        except Exception as e_fast:
            # Fallback al tokenizer lento
            try:
                enc = self.tok_slow.encode_plus(
                    text=text,
                    text_pair=pair,
                    truncation=True,
                    max_length=self.seq_max,
                    padding="max_length",
                    return_tensors="pt",
                )
            except Exception as e_slow:
                rid = None
                try:
                    idx = getattr(self, "_last_index", None)
                    if idx is not None and "id" in self.df.columns:
                        rid = self.df.iloc[idx]["id"]
                except Exception:
                    pass
                raise TypeError(
                    f"❌ Tokenización fallida | id={rid} | "
                    f"types: prompt={type(prompt).__name__}, resp={type(resp).__name__} | "
                    f"fast_err={e_fast} | slow_err={e_slow}"
                )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, i):
        self._last_index = i
        row = self.df.iloc[i]
        ex1 = self.encodePair(row["prompt"], row["response_a"])
        ex2 = self.encodePair(row["prompt"], row["response_b"])
        y = 1 if row.get("winner_model_a", 0) == 1 else 0
        tie = 1 if row.get("winner_tie", 0) == 1 else 0
        return ex1, ex2, torch.tensor(y, dtype=torch.float32), torch.tensor(tie, dtype=torch.float32)
