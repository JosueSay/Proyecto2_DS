import os, torch, numpy as np, pandas as pd, logging
from transformers import AutoTokenizer, AutoConfig
from model import CrossEncoder
import argparse

# Preferimos paralelismo del tokenizer en local (desactívalo si satura CPU)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
USE_SLOW = os.getenv("USE_SLOW_TOKENIZER", "0") == "1"

class PairRanker:
    def __init__(self, model_dir, max_len_prompt=256, max_len_resp=700):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("GPU: %s | CUDA %s", torch.cuda.get_device_name(0), torch.version.cuda)
        else:
            logging.info("CPU")

        # ---- Tokenizer (usa el guardado si existe)
        tok_dir = os.path.join(model_dir, "tokenizer")
        tok_src = tok_dir if os.path.isdir(tok_dir) else model_dir
        self.tok = AutoTokenizer.from_pretrained(tok_src, use_fast=not USE_SLOW)

        # ---- Modelo
        cfg = AutoConfig.from_pretrained(model_dir)
        self.model = CrossEncoder(cfg._name_or_path).to(self.device).eval()
        state_path = os.path.join(model_dir, "model.bin")
        self.model.load_state_dict(torch.load(state_path, map_location=self.device))

        # (Opcional) compilar para PyTorch 2.x; si da problemas, comenta esta línea
        # self.model = torch.compile(self.model)

        # ---- Longitudes
        tok_max = getattr(self.tok, "model_max_length", 512) or 512
        self.seq_max = min(int(max_len_prompt) + int(max_len_resp) + 3, tok_max if tok_max > 0 else 512)
        logging.info("Inferencia seq_max=%d | backbone=%s", self.seq_max, cfg._name_or_path)

        # AMP en inferencia (bf16 si hay GPU Ada/Lovelace; si prefieres fp16, quita dtype)
        self.use_amp = (self.device == "cuda")
        self.autocast_kwargs = dict(device_type="cuda", dtype=torch.bfloat16) if self.use_amp else dict(enabled=False)

    @torch.inference_mode()
    def predictOne(self, prompt, ra, rb):
        def enc(resp_text):
            e = self.tok.encode_plus(
                str(prompt), str(resp_text),
                truncation=True,
                max_length=self.seq_max,
                padding="max_length",
                return_tensors="pt"
            )
            return {k: v.to(self.device, non_blocking=True) for k, v in e.items()}

        encA, encB = enc(ra), enc(rb)

        if self.use_amp:
            with torch.amp.autocast(**self.autocast_kwargs):
                sa, sb = self.model(encA, encB)
        else:
            sa, sb = self.model(encA, encB)

        # Prob A vs B y empate heurístico
        delta = (sa - sb).item()
        pA = torch.sigmoid(torch.tensor(delta)).item()
        pB = 1.0 - pA
        pT = float(np.clip(0.5 - abs(delta)/6.0, 0.0, 0.5))  # ajusta si quieres menos empates
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
