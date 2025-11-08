import os, argparse, logging
import torch, numpy as np, pandas as pd
from transformers import AutoTokenizer
from model import CrossEncoder
from config_loader import loadYamlConfig, getValue

def readStrictCfg(model_dir: str, external_cfg: str | None) -> dict:
    # carga config externa o del checkpoint
    if external_cfg:
        return loadYamlConfig(external_cfg)
    local_cfg = os.path.join(model_dir, "train_config.yaml")
    if os.path.isfile(local_cfg):
        return loadYamlConfig(local_cfg)
    raise FileNotFoundError("no se encontró train_config.yaml en el modelo ni se pasó --config")

def pickTextColsStrict(df: pd.DataFrame, cfg: dict):
    # columnas obligatorias para inferencia
    try:
        p = getValue(cfg, "data.prompt_col")
        a = getValue(cfg, "data.respA_col")
        b = getValue(cfg, "data.respB_col")
    except KeyError as e:
        raise KeyError(f"falta en default.yaml: {e}. agrega data.prompt_col/respA_col/respB_col") from e
    need = {p, a, b}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"faltan columnas para inferencia en el csv: {miss}")
    return p, a, b

def seqMaxLenStrict(tok, max_len_prompt: int, max_len_resp: int) -> int:
    # calcula longitud máxima respetando el backbone
    tok_max = getattr(tok, "model_max_length", None)
    if not isinstance(tok_max, int) or tok_max <= 0:
        raise ValueError("tokenizer.model_max_length inválido; revisa el backbone/tokenizer")
    return min(int(max_len_prompt) + int(max_len_resp) + 3, tok_max)

def toDevice(batch: dict, device: str):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

class PairRanker:
    def __init__(self, model_dir: str, cfg: dict):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("gpu: %s | cuda %s", torch.cuda.get_device_name(0), torch.version.cuda)
        else:
            logging.info("cpu inferencia")

        # parámetros obligatorios desde yaml
        try:
            backbone = getValue(cfg, "pretrained_name")
            max_len_prompt = getValue(cfg, "max_len_prompt")
            max_len_resp = getValue(cfg, "max_len_resp")
            use_slow_tokenizer = getValue(cfg, "env.use_slow_tokenizer")
        except KeyError as e:
            raise KeyError(f"falta en default.yaml: {e}. agrega la clave requerida") from e

        # tokenizador
        tok_dir = os.path.join(model_dir, "tokenizer")
        tok_src = tok_dir if os.path.isdir(tok_dir) else backbone
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=not bool(use_slow_tokenizer))

        # modelo
        self.model = CrossEncoder(backbone, cfg).to(self.device_type).eval()

        state_path = os.path.join(model_dir, "model.bin")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"no se encontró {state_path}")
        state = torch.load(state_path, map_location=self.device_type)
        self.model.load_state_dict(state, strict=True)

        self.seq_max = seqMaxLenStrict(self.tokenizer, max_len_prompt, max_len_resp)
        logging.info("seq_max=%d | backbone=%s", self.seq_max, backbone)

        # amp bf16 solo si hay gpu
        self.use_amp = (self.device_type == "cuda")
        self.autocast_kwargs = dict(device_type="cuda", dtype=torch.bfloat16) if self.use_amp else dict(enabled=False)

        self.cfg = cfg

    def encodePair(self, prompt_text: str, resp_text: str):
        # codifica (prompt, response)
        return self.tokenizer.encode_plus(
            str(prompt_text), str(resp_text),
            truncation=True,
            max_length=self.seq_max,
            padding="max_length",
            return_tensors="pt"
        )

    @torch.inference_mode()
    def predictOne(self, prompt_text, response_a, response_b):
        enc_a = toDevice(self.encodePair(prompt_text, response_a), self.device_type)
        enc_b = toDevice(self.encodePair(prompt_text, response_b), self.device_type)
        if self.use_amp:
            with torch.amp.autocast(**self.autocast_kwargs):
                logits, _, _ = self.model(enc_a, enc_b)
        else:
            logits, _, _ = self.model(enc_a, enc_b)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return float(probs[0]), float(probs[1]), float(probs[2])  # A, B, TIE

    def predictCsv(self, test_csv_path: str, out_csv_path: str):
        df = pd.read_csv(test_csv_path)
        p_col, a_col, b_col = pickTextColsStrict(df, self.cfg)

        rows = []
        for idx, r in df.iterrows():
            pa, pb, pt = self.predictOne(r[p_col], r[a_col], r[b_col])
            rid = r["id"] if "id" in r else idx
            rows.append([rid, pa, pb, pt])

        sub = pd.DataFrame(rows, columns=["id", "winner_model_a", "winner_model_b", "winner_tie"])
        sub.to_csv(out_csv_path, index=False)
        logging.info("predicciones guardadas en: %s", out_csv_path)
        return out_csv_path

    @staticmethod
    def computeMetrics(frame: pd.DataFrame) -> pd.DataFrame:
        # calcula accuracy y macro-f1
        if "label" in frame.columns:
            y_true = frame["label"].to_numpy()
        else:
            cols = ["winner_model_a_y", "winner_model_b_y", "winner_tie_y"]
            if not set(cols).issubset(frame.columns):
                return pd.DataFrame([{"metric": "accuracy", "value": np.nan},
                                     {"metric": "macro_f1", "value": np.nan}])
            y_true = frame[cols].to_numpy().argmax(1)

        y_pred = frame[["winner_model_a_p", "winner_model_b_p", "winner_tie_p"]].to_numpy().argmax(1)
        acc = float((y_true == y_pred).mean())

        scores = []
        for ci in range(3):
            tp = int(((y_pred == ci) & (y_true == ci)).sum())
            fp = int(((y_pred == ci) & (y_true != ci)).sum())
            fn = int(((y_pred != ci) & (y_true == ci)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            scores.append(f1)
        macro_f1 = float(np.mean(scores))
        return pd.DataFrame([{"metric": "accuracy", "value": acc},
                             {"metric": "macro_f1", "value": macro_f1}])

    def maybeMetrics(self, labeled_csv_path: str, preds_csv_path: str, metrics_out_path: str | None = None):
        gt = pd.read_csv(labeled_csv_path)
        pr = pd.read_csv(preds_csv_path)

        cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
        merged = gt.merge(pr[cols], on="id", suffixes=("_y", "_p"), how="inner")
        metrics = self.computeMetrics(merged)

        if metrics_out_path:
            os.makedirs(os.path.dirname(metrics_out_path), exist_ok=True)
            metrics.to_csv(metrics_out_path, index=False)
            logging.info("métricas guardadas en: %s", metrics_out_path)
        else:
            logging.info("métricas:\n%s", metrics.to_string(index=False))
        return metrics

    @classmethod
    def load(cls, model_dir: str, cfg: dict):
        return cls(model_dir, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics_out", default=None)
    parser.add_argument("--config", default=None, help="ruta a un yaml alterno si no existe train_config.yaml")
    args = parser.parse_args()

    cfg = readStrictCfg(args.model_dir, args.config)
    ranker = PairRanker.load(args.model_dir, cfg)

    out_path = ranker.predictCsv(args.test, args.out)
    ranker.maybeMetrics(args.test, out_path, args.metrics_out)
    print(f"saved to {out_path}")
