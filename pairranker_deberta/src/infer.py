import os, torch, numpy as np, pandas as pd, logging
from transformers import AutoTokenizer, AutoConfig
from model import CrossEncoder
import argparse

class PairRanker:
    def __init__(self, model_dir, max_len_prompt=256, max_len_resp=700):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        tok_dir = os.path.join(model_dir, "tokenizer")
        self.tok = AutoTokenizer.from_pretrained(tok_dir)
        cfg = AutoConfig.from_pretrained(model_dir)
        self.model = CrossEncoder(cfg._name_or_path)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "model.bin"), map_location="cpu"))
        self.model.eval()
        tok_max = getattr(self.tok, "model_max_length", 512)
        self.seq_max = min(int(max_len_prompt) + int(max_len_resp) + 3, tok_max)
        logging.info("Inferencia con seq_max=%d, backbone=%s", self.seq_max, cfg._name_or_path)

    @torch.no_grad()
    def predictOne(self, prompt, ra, rb):
        def enc(resp_text):
            return self.tok.encode_plus(
                str(prompt), str(resp_text),
                truncation=True,
                max_length=self.seq_max,
                padding="max_length",
                return_tensors="pt"
            )
        sa, sb = self.model(enc(ra), enc(rb))
        pA = torch.sigmoid(sa - sb).item()
        pB = 1 - pA
        delta = abs((sa - sb).item())
        pT = float(np.clip(0.5 - delta/6.0, 0.0, 0.5))
        z = pA + pB + pT
        return pA/z, pB/z, pT/z

    def predictCsv(self, test_csv, out_csv):
        df = pd.read_csv(test_csv)
        rows = []
        for _, r in df.iterrows():
            pa, pb, pt = self.predictOne(r["prompt"], r["response_a"], r["response_b"])
            rows.append([r["id"], pa, pb, pt])
        sub = pd.DataFrame(rows, columns=["id","winner_model_a","winner_model_b","winner_tie"])
        sub.to_csv(out_csv, index=False)
        logging.info("✅ Guardado submission: %s", out_csv)
        return out_csv

    @classmethod
    def load(cls, model_dir, max_len_prompt=256, max_len_resp=700):
        return cls(model_dir, max_len_prompt, max_len_resp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len_prompt", type=int, default=256)
    ap.add_argument("--max_len_resp", type=int, default=700)
    args = ap.parse_args()

    ranker = PairRanker.load(args.model_dir, args.max_len_prompt, args.max_len_resp)
    path = ranker.predictCsv(args.test, args.out)
    print(f"Saved to {path}")