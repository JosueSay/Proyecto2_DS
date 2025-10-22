import pandas as pd, numpy as np, torch, ast
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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

class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, model_name: str, max_len_prompt=256, max_len_resp=700):
        self.df = df.reset_index(drop=True)
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.maxp, self.maxr = int(max_len_prompt), int(max_len_resp)
        tok_max = getattr(self.tok, "model_max_length", 512)
        self.seq_max = min(self.maxp + self.maxr + 3, tok_max)

    def __len__(self):
        return len(self.df)

    def encodePair(self, prompt, resp):
        prompt = maybeJoin(prompt)
        resp   = maybeJoin(resp)
        enc = self.tok.encode_plus(
            str(prompt), str(resp),
            truncation=True,
            max_length=self.seq_max,
            padding="max_length",
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __getitem__(self, i):
        row = self.df.iloc[i]
        ex1 = self.encodePair(row["prompt"], row["response_a"])
        ex2 = self.encodePair(row["prompt"], row["response_b"])
        y = 1 if row.get("winner_model_a", 0) == 1 else 0
        tie = 1 if row.get("winner_tie", 0) == 1 else 0
        return ex1, ex2, torch.tensor(y, dtype=torch.float32), torch.tensor(tie, dtype=torch.float32)
