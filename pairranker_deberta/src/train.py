import os, yaml, random, numpy as np, torch, logging, math
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from data import PairDataset
from model import CrossEncoder
from losses import btLoss
import pandas as pd
from tqdm import tqdm
from torch import amp
from transformers import AutoTokenizer
import argparse

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collateBatch(batch):
    encA_list, encB_list, y_list, tie_list = zip(*batch)
    def stackEnc(lst):
        keys = lst[0].keys()
        return {k: torch.stack([ex[k] for ex in lst], dim=0) for k in keys}
    return stackEnc(encA_list), stackEnc(encB_list), torch.stack(y_list), torch.stack(tie_list)

def printRunConfig(cfg, device, model_name, seq_max, ds_len, steps_per_epoch):
    logging.info("=========== RUN CONFIG ===========")
    logging.info("Model name\t: %s", model_name)
    logging.info("Device\t: %s", device)
    if device == "cuda":
        logging.info("GPU\t: %s", torch.cuda.get_device_name(0))
        logging.info("CUDA version\t: %s", torch.version.cuda)
        logging.info("Compute capability\t: %s", ".".join(map(str, torch.cuda.get_device_capability())))
        logging.info("Amp enabled\t: True")
    else:
        logging.info("Amp enabled\t: False (CPU)")
    logging.info("Tokenizer seq_max\t: %d tokens", seq_max)
    logging.info("Dataset size\t: %d ejemplos", ds_len)
    logging.info("Batch size\t: %d", cfg["batch_size"])
    logging.info("Epochs\t: %d", cfg["num_epochs"])
    logging.info("Steps/epoch\t: %d", steps_per_epoch)
    logging.info("Learning rate\t: %.6f", cfg["lr"])
    logging.info("Weight decay\t: %.6f", cfg["weight_decay"])
    logging.info("Warmup ratio\t: %.4f", cfg["warmup_ratio"])
    logging.info("Seed\t: %d", cfg["seed"])
    logging.info("Use ordinal tie\t: %s", cfg.get("use_ordinal_tie", True))
    logging.info("=================================")

def trainModel(train_csv, out_dir, cfg_path="configs/default.yaml"):
    # logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.makedirs(out_dir, exist_ok=True)

    # leer config y tipos
    cfg = yaml.safe_load(open(cfg_path))
    cfg["lr"] = float(cfg["lr"])
    cfg["weight_decay"] = float(cfg["weight_decay"])
    cfg["warmup_ratio"] = float(cfg["warmup_ratio"])
    cfg["batch_size"] = int(cfg["batch_size"])
    cfg["num_epochs"] = int(cfg["num_epochs"])
    cfg["max_len_prompt"] = int(cfg["max_len_prompt"])
    cfg["max_len_resp"] = int(cfg["max_len_resp"])
    cfg["seed"] = int(cfg["seed"])

    setSeed(cfg["seed"])

    # device + TF32 moderna
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'

    # ===== Datos: fuerza texto limpio y consistente =====
    df = pd.read_csv(train_csv)
    required_cols = ["prompt", "response_a", "response_b", "winner_model_a", "winner_model_b", "winner_tie"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {train_csv}: {missing}")

    text_cols = ["prompt", "response_a", "response_b"]
    # fillna ANTES de astype(str) para no convertir NaN->"nan"
    df[text_cols] = df[text_cols].fillna("")
    df[text_cols] = df[text_cols].astype(str)

    # log de dtypes y una muestra
    logging.info("dtypes texto: %s", dict(df[text_cols].dtypes))
    logging.info("muestras prompt/resp (primeras 2 filas):")
    for idx in range(min(2, len(df))):
        logging.info("id=%s | prompt-type=%s | respA-type=%s | respB-type=%s",
                     str(df.loc[idx, "id"]) if "id" in df.columns else f"row{idx}",
                     type(df.loc[idx, "prompt"]).__name__,
                     type(df.loc[idx, "response_a"]).__name__,
                     type(df.loc[idx, "response_b"]).__name__)

    ds = PairDataset(df, cfg["model_name"], cfg["max_len_prompt"], cfg["max_len_resp"])
    seq_max = ds.seq_max

    num_workers = int(cfg.get("num_workers", 2))
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collateBatch,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0)
    )

    # modelo/opt/scheduler
    model = CrossEncoder(cfg["model_name"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = max(1, cfg["num_epochs"] * len(dl))
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)
    sch = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    use_amp = bool(cfg.get("amp", True) and device == "cuda")
    scaler = amp.GradScaler('cuda', enabled=use_amp)

    printRunConfig(cfg, device, cfg["model_name"], seq_max, len(ds), math.ceil(len(ds) / cfg["batch_size"]))
    logging.info("Total params      : %.2f M", total_params / 1e6)
    logging.info("Trainable params  : %.2f M", trainable_params / 1e6)

    # loop
    model.train()
    for ep in range(cfg["num_epochs"]):
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{cfg['num_epochs']}")
        epoch_loss = 0.0
        for step, (encA, encB, y, tie) in enumerate(pbar, start=1):
            encA = {k: v.to(device, non_blocking=True) for k, v in encA.items()}
            encB = {k: v.to(device, non_blocking=True) for k, v in encB.items()}
            y, tie = y.to(device, non_blocking=True), tie.to(device, non_blocking=True)

            try:
                with amp.autocast('cuda', enabled=use_amp):
                    sa, sb = model(encA, encB)
                    loss, _ = btLoss(sa, sb, y, tie if cfg.get("use_ordinal_tie", True) else None)

                # validar pérdida
                if not torch.isfinite(loss):
                    logging.warning("💥 loss no finita (NaN/Inf). Saltando batch step=%d", step)
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
            except Exception as e:
                logging.error("Fallo en batch step=%d (ep=%d). Error: %s", step, ep+1, repr(e))
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                raise

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()

            l = loss.item()
            epoch_loss += l
            pbar.set_postfix_str(f"loss={l:.4f}")

        logging.info("Epoch %d/%d | mean loss: %.6f", ep+1, cfg["num_epochs"], epoch_loss / step)

    # guardar artefactos
    torch.save(model.state_dict(), os.path.join(out_dir, "model.bin"))
    model.backbone.config.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(cfg["model_name"]).save_pretrained(os.path.join(out_dir, "tokenizer"))
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "train_config.yaml"), "w"))
    logging.info("✅ Modelo guardado en: %s", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)   # data/train.csv
    ap.add_argument("--out", required=True)     # runs/deberta_pairranker
    ap.add_argument("--cfg", default="configs/default.yaml")
    args = ap.parse_args()
    trainModel(args.train, args.out, args.cfg)
