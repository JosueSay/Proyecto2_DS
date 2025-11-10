import os
import sys
import json
import torch
import pandas as pd
from typing import Dict, Any

from .state import getDevice
from .paths import repoRoot
from .runs import getLatestRun

# importar implementación del modelo entrenado
PAIR_SRC = os.path.join(repoRoot(), "m_pair-ranker", "src")
if PAIR_SRC not in sys.path:
    sys.path.append(PAIR_SRC)

from pairranker.models.cross_encoder import CrossEncoder  # noqa
from transformers import AutoTokenizer  # noqa

LABELS = ["A", "B", "TIE"]

def _loadArtifacts(model_key: str) -> Dict[str, Any]:
    run = getLatestRun(model_key)
    run_dir = run["results_dir"]
    cfg_path = os.path.join(run_dir, "train_config.yaml")
    if not os.path.exists(cfg_path):
        # fallback a config.json si viene del HF
        cfg_path = os.path.join(run_dir, "config.json")
    tok_dir = os.path.join(run_dir, "tokenizer")
    bin_path = os.path.join(run_dir, "model.bin")
    if not (os.path.isdir(tok_dir) and os.path.isfile(bin_path)):
        raise FileNotFoundError(f"artefactos incompletos en {run_dir}")
    return {"run": run, "cfg_path": cfg_path, "tok_dir": tok_dir, "bin_path": bin_path}

def _readCfg(cfg_path: str) -> Dict[str, Any]:
    if cfg_path.endswith(".yaml"):
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _buildModel(cfg: Dict[str, Any], device: str) -> CrossEncoder:
    pretrained = cfg["model"]["pretrained_name"]
    dropout = float(cfg["model"].get("dropout", 0.1))
    grad_ckpt = bool(cfg["model"].get("grad_checkpointing", False))
    compile_flag = bool(cfg["model"].get("compile", False))
    model = CrossEncoder(pretrained_name=pretrained,
                         cfg=cfg,
                         dropout=dropout,
                         grad_checkpointing=grad_ckpt,
                         compile_flag=compile_flag).to(device)
    return model

def _tokenizeBatch(df_clean: pd.DataFrame, tok: AutoTokenizer, cfg: Dict[str, Any], device: str):
    max_p = int(cfg["lengths"]["max_len_prompt"])
    max_r = int(cfg["lengths"]["max_len_resp"])
    enc_a = tok(
        df_clean["prompt_clean"].tolist(),
        df_clean["response_a_clean"].tolist(),
        truncation="longest_first",
        max_length=max_p + max_r + 3,
        padding=True,
        return_tensors="pt",
    )
    enc_b = tok(
        df_clean["prompt_clean"].tolist(),
        df_clean["response_b_clean"].tolist(),
        truncation="longest_first",
        max_length=max_p + max_r + 3,
        padding=True,
        return_tensors="pt",
    )
    enc_a = {k: v.to(device) for k, v in enc_a.items()}
    enc_b = {k: v.to(device) for k, v in enc_b.items()}
    return enc_a, enc_b

def predictBatch(df_clean: pd.DataFrame, model_key: str) -> pd.DataFrame:
    device = getDevice()
    arts = _loadArtifacts(model_key)
    cfg = _readCfg(arts["cfg_path"])
    tok = AutoTokenizer.from_pretrained(arts["tok_dir"])
    model = _buildModel(cfg, device)
    sd = torch.load(arts["bin_path"], map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    enc_a, enc_b = _tokenizeBatch(df_clean, tok, cfg, device)
    with torch.no_grad(), torch.amp.autocast(device_type=("cuda" if device == "cuda" else "cpu"), enabled=(device == "cuda")):
        out = model(enc_a, enc_b)  # dict con "logits"
        logits = out["logits"] if isinstance(out, dict) else out
        probs = torch.softmax(logits, dim=-1)

    pred_idx = probs.argmax(dim=-1).tolist()
    df_out = pd.DataFrame({
        "id": df_clean["id"].tolist(),
        "prob_A": probs[:, 0].float().cpu().numpy(),
        "prob_B": probs[:, 1].float().cpu().numpy(),
        "prob_TIE": probs[:, 2].float().cpu().numpy(),
        "pred_label": [LABELS[i] for i in pred_idx],
    })
    return df_out

def predictSingle(df_clean_row: pd.DataFrame, model_key: str) -> Dict[str, Any]:
    out = predictBatch(df_clean_row, model_key)
    return out.iloc[0].to_dict()
