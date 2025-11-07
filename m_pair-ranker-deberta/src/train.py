import os, yaml, random, numpy as np, torch, logging, math
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from data import PairDataset
from model import CrossEncoder
from losses import btLoss
import pandas as pd
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import argparse

def setSeed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def collateBatch(batch):
    encA_list, encB_list, y_list, tie_list = zip(*batch)
    def stackEnc(lst):
        keys = lst[0].keys()
        return {k: torch.stack([ex[k] for ex in lst], dim=0) for k in keys}
    return stackEnc(encA_list), stackEnc(encB_list), torch.stack(y_list), torch.stack(tie_list)

def printRunConfig(cfg, device, model_name, seq_max, ds_len, steps_per_epoch, use_amp):
    logging.info("=========== RUN CONFIG ===========")
    logging.info("Model name\t: %s", model_name)
    logging.info("Device\t: %s", device)
    if device == "cuda":
        logging.info("GPU\t: %s", torch.cuda.get_device_name(0))
        logging.info("CUDA version\t: %s", torch.version.cuda)
        logging.info("Compute capability\t: %s", ".".join(map(str, torch.cuda.get_device_capability())))
    logging.info("Amp enabled\t: %s", use_amp)
    logging.info("Tokenizer seq_max\t: %d tokens", seq_max)
    logging.info("Dataset size\t: %d ejemplos", ds_len)
    logging.info("Batch size\t: %d", cfg["batch_size"])
    logging.info("Grad accum\t: %d", cfg.get("grad_accum", 1))
    logging.info("Epochs\t: %d", cfg["num_epochs"])
    logging.info("Steps/epoch\t: %d", steps_per_epoch)
    logging.info("Learning rate\t: %.6f", cfg["lr"])
    logging.info("Weight decay\t: %.6f", cfg["weight_decay"])
    logging.info("Warmup ratio\t: %.4f", cfg["warmup_ratio"])
    logging.info("Seed\t: %d", cfg["seed"])
    logging.info("Use ordinal tie\t: %s", cfg.get("use_ordinal_tie", True))
    logging.info("=================================")

def trainModel(train_csv, out_dir, cfg_path="configs/default.yaml"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    os.makedirs(out_dir, exist_ok=True)

    # CFG
    cfg = yaml.safe_load(open(cfg_path))
    for k in ("lr","weight_decay","warmup_ratio"): cfg[k] = float(cfg[k])
    for k in ("batch_size","num_epochs","max_len_prompt","max_len_resp","seed"): cfg[k] = int(cfg[k])
    cfg["grad_accum"] = int(cfg.get("grad_accum", 1))

    setSeed(cfg["seed"])

    # Device + TF32 actual
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Datos
    df = pd.read_csv(train_csv)
    required_cols = ["prompt","response_a","response_b","winner_model_a","winner_model_b","winner_tie"]
    miss = [c for c in required_cols if c not in df.columns]
    if miss: raise ValueError(f"Faltan columnas en {train_csv}: {miss}")

    text_cols = ["prompt","response_a","response_b"]
    df[text_cols] = df[text_cols].fillna("").astype(str)

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

    # DataLoader seguro (Colab/WSL): sin persistentes por defecto
    num_workers = int(cfg.get("num_workers", 0))
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collateBatch,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=False
    )

    # Modelo/opt/sched
    model = CrossEncoder(cfg["model_name"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_steps = max(1, cfg["num_epochs"] * len(dl) // max(1, cfg["grad_accum"]))
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)
    sch = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    use_amp = bool(cfg.get("amp", True) and device == "cuda")
    # En Ada (4070 Ti) puedes usar bf16:
    autocast_kwargs = dict(device_type="cuda", dtype=torch.bfloat16) if use_amp else dict(enabled=False)
    scaler = GradScaler('cuda', enabled=use_amp)

    printRunConfig(cfg, device, cfg["model_name"], seq_max, len(ds),
                   math.ceil(len(ds)/cfg["batch_size"]), use_amp)
    logging.info("Total params      : %.2f M", total_params/1e6)
    logging.info("Trainable params  : %.2f M", trainable_params/1e6)

    # Loop
    model.train()
    grad_accum = max(1, cfg["grad_accum"])
    opt.zero_grad(set_to_none=True)

    for ep in range(cfg["num_epochs"]):
        pbar = tqdm(dl, desc=f"epoch {ep+1}/{cfg['num_epochs']}")
        epoch_loss = 0.0
        for step, (encA, encB, y, tie) in enumerate(pbar, start=1):
            encA = {k: v.to(device, non_blocking=True) for k, v in encA.items()}
            encB = {k: v.to(device, non_blocking=True) for k, v in encB.items()}
            y, tie = y.to(device, non_blocking=True), tie.to(device, non_blocking=True)

            try:
                with autocast(**autocast_kwargs):
                    sa, sb = model(encA, encB)
                    loss, _ = btLoss(sa, sb, y, tie if cfg.get("use_ordinal_tie", True) else None)
                    loss = loss / grad_accum

                if not torch.isfinite(loss):
                    logging.warning("💥 loss no finita (NaN/Inf). Saltando batch step=%d", step)
                    opt.zero_grad(set_to_none=True)
                    if device == "cuda": torch.cuda.empty_cache()
                    continue

            except RuntimeError as re:
                if "out of memory" in str(re).lower() and device == "cuda":
                    logging.error("⚠️ OOM en step=%d. Limpiando y continuando.", step)
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            scaler.scale(loss).backward()

            if step % grad_accum == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()

            l = loss.item() * grad_accum
            epoch_loss += l
            pbar.set_postfix_str(f"loss={l:.4f}")

        mean_loss = epoch_loss / max(1, step)
        logging.info("Epoch %d/%d | mean loss: %.6f", ep+1, cfg["num_epochs"], mean_loss)

    # Guardado
    torch.save(model.state_dict(), os.path.join(out_dir, "model.bin"))
    try:
        model.backbone.config.save_pretrained(out_dir)
    except Exception:
        logging.warning("No se pudo guardar config desde model.backbone; se omite.")
    AutoTokenizer.from_pretrained(cfg["model_name"]).save_pretrained(os.path.join(out_dir, "tokenizer"))
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "train_config.yaml"), "w"))
    logging.info("✅ Modelo guardado en: %s", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="configs/default.yaml")
    args = ap.parse_args()
    trainModel(args.train, args.out, args.cfg)
