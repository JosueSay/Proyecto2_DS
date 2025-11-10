import os
import sys
import json
import pandas as pd
from typing import Dict, Any

from .state import getDevice, getAppConfig
from .paths import repoRoot
from .runs import getLatestRun

PAIR_SRC = os.path.join(repoRoot(), "m_pair-ranker", "src")
if PAIR_SRC not in sys.path:
    sys.path.append(PAIR_SRC)


def safeImportTorch():
    try:
        import torch
        return torch
    except Exception:
        return None


def safeImportTokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer
    except Exception:
        return None


def tryImportCrossEncoder():
    try:
        from pairranker.models.cross_encoder import CrossEncoder
        return CrossEncoder
    except Exception:
        return None


LABELS = ["A", "B", "TIE"]


def loadArtifacts(model_key: str) -> Dict[str, Any]:
    cfg = getAppConfig()
    run = getLatestRun(model_key, cfg)
    if not run or not run.get("results_dir"):
        raise FileNotFoundError(f"no hay run válido para el modelo '{model_key}'")

    run_dir = run["results_dir"]
    cfg_path = os.path.join(run_dir, "train_config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(run_dir, "config.json")

    tok_dir = os.path.join(run_dir, "tokenizer")
    bin_path = os.path.join(run_dir, "model.bin")

    if not os.path.isdir(tok_dir):
        raise FileNotFoundError(f"no existe tokenizer en {tok_dir}")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"no existe model.bin en {bin_path}")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"no existe config en {cfg_path}")

    return {"run": run, "cfg_path": cfg_path, "tok_dir": tok_dir, "bin_path": bin_path}


def readCfg(cfg_path: str) -> Dict[str, Any]:
    if cfg_path.endswith(".yaml") or cfg_path.endswith(".yml"):
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def buildModel(cfg: Dict[str, Any], device: str):
    CrossEncoder = tryImportCrossEncoder()
    if CrossEncoder is None:
        return None
    pretrained = cfg["model"]["pretrained_name"]
    dropout = float(cfg["model"].get("dropout", 0.1))
    grad_ckpt = bool(cfg["model"].get("grad_checkpointing", False))
    compile_flag = bool(cfg["model"].get("compile", False))
    model = CrossEncoder(
        pretrained_name=pretrained,
        cfg=cfg,
        dropout=dropout,
        grad_checkpointing=grad_ckpt,
        compile_flag=compile_flag,
    )
    # mover a device más adelante cuando torch esté confirmado
    return model


def tokenizeBatch(df_clean: pd.DataFrame, tok, cfg: Dict[str, Any], device: str):
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

    torch = safeImportTorch()
    if torch is not None:
        enc_a = {k: v.to(device) for k, v in enc_a.items()}
        enc_b = {k: v.to(device) for k, v in enc_b.items()}

    return enc_a, enc_b


def placeholderOutput(df_clean: pd.DataFrame) -> pd.DataFrame:
    # salida neutra cuando torch/model no están disponibles
    n = len(df_clean)
    probs_a = [0.3333] * n
    probs_b = [0.3333] * n
    probs_t = [0.3333] * n
    preds = ["TIE"] * n
    ids = df_clean["id"].tolist() if "id" in df_clean.columns else list(range(n))
    return pd.DataFrame(
        {
            "id": ids,
            "prob_A": probs_a,
            "prob_B": probs_b,
            "prob_TIE": probs_t,
            "pred_label": preds,
        }
    )


def predictBatch(df_clean: pd.DataFrame, model_key: str) -> pd.DataFrame:
    torch = safeImportTorch()
    AutoTokenizer = safeImportTokenizer()

    # si no hay torch o tokenizer usable, devolvemos placeholder
    if torch is None or AutoTokenizer is None:
        return placeholderOutput(df_clean)

    device = getDevice()  # "cuda" si torch.cuda.is_available(), si no "cpu"
    try:
        arts = loadArtifacts(model_key)
        cfg = readCfg(arts["cfg_path"])
    except Exception:
        return placeholderOutput(df_clean)

    # construir modelo
    model = buildModel(cfg, device)
    if model is None:
        return placeholderOutput(df_clean)

    # mover modelo al device solo si torch importó bien
    try:
        model = model.to(device)
    except Exception:
        return placeholderOutput(df_clean)

    # cargar tokenizer y pesos
    try:
        tok = AutoTokenizer.from_pretrained(arts["tok_dir"])
    except Exception:
        return placeholderOutput(df_clean)

    try:
        state_dict = torch.load(arts["bin_path"], map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception:
        return placeholderOutput(df_clean)

    # tokenizar y ejecutar
    try:
        enc_a, enc_b = tokenizeBatch(df_clean, tok, cfg, device)
        use_cuda = device == "cuda"
        with torch.no_grad(), torch.amp.autocast(
            device_type=("cuda" if use_cuda else "cpu"), enabled=use_cuda
        ):
            out = model(enc_a, enc_b)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.softmax(logits, dim=-1)

        pred_idx = probs.argmax(dim=-1).tolist()
        ids = df_clean["id"].tolist() if "id" in df_clean.columns else list(range(len(df_clean)))
        df_out = pd.DataFrame(
            {
                "id": ids,
                "prob_A": probs[:, 0].float().cpu().numpy(),
                "prob_B": probs[:, 1].float().cpu().numpy(),
                "prob_TIE": probs[:, 2].float().cpu().numpy(),
                "pred_label": [LABELS[i] for i in pred_idx],
            }
        )
        return df_out
    except Exception:
        return placeholderOutput(df_clean)


def predictSingle(df_clean_row: pd.DataFrame, model_key: str) -> Dict[str, Any]:
    out = predictBatch(df_clean_row, model_key)
    return out.iloc[0].to_dict()
