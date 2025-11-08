import os, math, time, yaml, argparse, logging, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from tqdm.auto import tqdm
import contextlib

def make_sdpa_ctx():
    # usa flash-attention si está disponible; de lo contrario, no aplica ningún contexto
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    except Exception:
        return contextlib.nullcontext()

from data import PairDataset, collateFn
from model import CrossEncoder
from losses import makeLoss
from config_loader import loadYamlConfig, getValue

# === utilidades básicas ===
def setSeed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensureDirs(path):
    os.makedirs(path, exist_ok=True)

def nowStr():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def getLr(optimizer):
    # toma el lr del primer param group
    return optimizer.param_groups[0]["lr"]

def getGpuMem():
    # muestra uso de VRAM actual y máximo
    if torch.cuda.is_available():
        used = torch.cuda.max_memory_allocated() / (1024**2)
        cur = torch.cuda.memory_allocated() / (1024**2)
        return f"{cur:.0f}MB (max {used:.0f}MB)"
    return "cpu"


# === validación ===
@torch.inference_mode()
def runValidation(model, loader, device, use_amp, epoch_idx, logs):
    all_probs, all_y, losses, steps = [], [], 0.0, 0
    loss_fn = makeLoss(logs["cfg"])
    amp_kwargs = dict(device_type="cuda", dtype=torch.bfloat16) if use_amp and device == "cuda" else dict(device_type=device, enabled=False)

    pbar = tqdm(loader, total=len(loader), desc=f"val {epoch_idx}", leave=False, dynamic_ncols=True)
    for encA, encB, y, _ in pbar:
        encA = {k: v.to(device, non_blocking=True) for k, v in encA.items()}
        encB = {k: v.to(device, non_blocking=True) for k, v in encB.items()}
        y = y.to(device, non_blocking=True)
        with autocast(**amp_kwargs):
            out = model(encA, encB)
            loss = loss_fn(out, {"label": y})
            probs = torch.softmax(out[0], dim=-1)
        losses += float(loss.detach().cpu()); steps += 1
        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        if steps % 10 == 0:
            pbar.set_postfix(loss=f"{losses/steps:.4f}")

    if steps == 0:
        return dict(val_loss=np.nan, val_acc=np.nan, macro_f1=np.nan, entropy=np.nan, dist=[np.nan]*3, conf=None, per=None)

    probs_np = np.concatenate(all_probs, 0)
    y_np = np.concatenate(all_y, 0)

    acc, macro_f1, ent, dist, conf = classMetrics(probs_np, y_np)
    per = perClassMetrics(conf)

    reports_dir = logs["reports_dir"]
    # predicciones fila a fila
    pd.DataFrame({
        "p_A": probs_np[:,0], "p_B": probs_np[:,1], "p_TIE": probs_np[:,2],
        "y_true": y_np.astype(int), "y_pred": probs_np.argmax(1).astype(int),
    }).to_csv(os.path.join(reports_dir, f"val_predictions_epoch_{epoch_idx:02d}.csv"), index=False)
    # matriz de confusión
    pd.DataFrame(conf, columns=["pred_A","pred_B","pred_TIE"]).assign(true=["A","B","TIE"]).to_csv(
        os.path.join(reports_dir, f"val_confusion_epoch_{epoch_idx:02d}.csv"), index=False
    )
    # métricas por clase
    pd.DataFrame([
        {"class":"A", **per["A"]},
        {"class":"B", **per["B"]},
        {"class":"TIE", **per["TIE"]},
    ]).to_csv(os.path.join(reports_dir, f"class_report_epoch_{epoch_idx:02d}.csv"), index=False)

    return dict(
        val_loss=losses/steps, val_acc=acc, macro_f1=macro_f1,
        entropy=ent, dist=dist, conf=conf, per=per
    )


def perClassMetrics(conf_mat):
    # precisión, recall y f1 por clase
    out, names = {}, ["A","B","TIE"]
    for c, name in enumerate(names):
        tp = int(conf_mat[c, c])
        fp = int(conf_mat[:, c].sum() - tp)
        fn = int(conf_mat[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[name] = dict(precision=prec, recall=rec, f1=f1)
    return out


def classMetrics(probs_np, y_np):
    # métricas agregadas y matriz de confusión 3x3
    y_pred = probs_np.argmax(1)
    acc = float((y_pred == y_np).mean())
    f1s, mat = [], np.zeros((3,3), dtype=int)
    for t, p in zip(y_np, y_pred): mat[t, p] += 1
    for c in range(3):
        tp = int(((y_pred == c) & (y_np == c)).sum())
        fp = int(((y_pred == c) & (y_np != c)).sum())
        fn = int(((y_pred != c) & (y_np == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    ent = float((-np.sum(probs_np * np.log(probs_np + 1e-8), axis=1)).mean())
    dist = probs_np.mean(0).tolist()
    return acc, macro_f1, ent, dist, mat


def collapseAlert(val_acc, ent, dist, tol=0.05):
    # detecta colapso tipo 33/33/33 (pred uniformes)
    uniform = all(abs(d - 1/3) <= tol for d in dist)
    return uniform and (val_acc < 0.36) and (ent > 0.95)


# === entrenamiento principal ===
def trainModel(train_csv, out_dir, cfg_path="m_pair-ranker-deberta/configs/default.yaml", valid_csv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ensureDirs(out_dir)

    cfg = loadYamlConfig(cfg_path)
    setSeed(getValue(cfg, "seed"))
    step_interval = getValue(cfg, "logging.step_interval")

    reports_dir = getValue(cfg, "logging.reports_dir")
    ensureDirs(reports_dir)

    step_csv   = os.path.join(reports_dir, getValue(cfg, "logging.step_csv"))
    epoch_csv  = os.path.join(reports_dir, getValue(cfg, "logging.epoch_csv"))
    alerts_csv = os.path.join(reports_dir, getValue(cfg, "logging.alerts_csv"))
    for p, cols in [
        (step_csv,  ["time","epoch","step","loss","lr"]),
        (epoch_csv, [
            "time","epoch","mean_train_loss","val_loss","val_acc","macro_f1","entropy",
            "dist_A","dist_B","dist_TIE",
            "precision_A","recall_A","f1_A",
            "precision_B","recall_B","f1_B",
            "precision_TIE","recall_TIE","f1_TIE","steps"
        ]),
        (alerts_csv,["time","epoch","type","detail"]),
    ]:
        if not os.path.exists(p): pd.DataFrame(columns=cols).to_csv(p, index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("gpu: %s | cuda %s", torch.cuda.get_device_name(0), torch.version.cuda)

    # datasets y loaders
    df_tr = pd.read_csv(train_csv)
    ds_tr = PairDataset(df_tr, cfg)
    
    nw  = int(getValue(cfg, "num_workers"))
    pf_tr = int(getValue(cfg, "dataloader.prefetch_factor_train"))
    pf_va = int(getValue(cfg, "dataloader.prefetch_factor_val"))
    pw  = bool(getValue(cfg, "dataloader.persistent_workers"))
    pin = bool(getValue(cfg, "data.pin_memory"))

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(getValue(cfg, "batch_size")),
        shuffle=bool(getValue(cfg, "data.shuffle")),
        collate_fn=collateFn,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(pw and nw > 0),
        prefetch_factor=(pf_tr if nw > 0 else None),
    )

    dl_va = None
    if valid_csv:
        df_va = pd.read_csv(valid_csv)
        ds_va = PairDataset(df_va, cfg)
        nw_va = max(0, nw // 2)
        dl_va = DataLoader(
            ds_va,
            batch_size=int(getValue(cfg, "data.val_batch_size")),
            shuffle=False,
            collate_fn=collateFn,
            num_workers=nw_va,
            pin_memory=pin,
            persistent_workers=(pw and nw_va > 0),
            prefetch_factor=(pf_va if nw_va > 0 else None),
        )

    model = CrossEncoder(getValue(cfg, "pretrained_name"), cfg).to(device)
    tot_p = sum(p.numel() for p in model.parameters())
    trn_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("params(M)=%.2f (trainable %.2f) | seq_max=%d", tot_p/1e6, trn_p/1e6, dl_tr.dataset.seq_max)

    # optimizador y scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=float(getValue(cfg, "lr")), weight_decay=float(getValue(cfg, "weight_decay")))
    steps_per_ep = math.ceil(len(dl_tr.dataset)/int(getValue(cfg, "batch_size")))
    total_steps  = max(1, int(getValue(cfg, "epochs")) * steps_per_ep // max(1, int(getValue(cfg, "grad_accum"))))
    warmup_steps = int(float(getValue(cfg, "warmup_ratio")) * total_steps)
    sched_name = str(getValue(cfg, "scheduler")).lower()
    if sched_name == "cosine":
        sch = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    elif sched_name == "linear":
        from transformers import get_linear_schedule_with_warmup
        sch = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    else:
        raise ValueError("scheduler debe ser 'cosine' o 'linear'")

    amp_mode = str(getValue(cfg, "amp")).lower()
    use_amp  = (device=="cuda") and (amp_mode != "false")
    if use_amp:
        if amp_mode == "bf16":
            amp_kwargs = dict(device_type="cuda", dtype=torch.bfloat16)
        elif amp_mode == "fp16":
            amp_kwargs = dict(device_type="cuda", dtype=torch.float16)
    else:
        amp_kwargs = dict(device_type=device, enabled=False)

    scaler = GradScaler(device if device=="cuda" else "cpu", enabled=(use_amp and amp_mode=="fp16"))

    loss_fn = makeLoss(cfg)
    best_metric, bad_epochs = None, 0
    start = time.time()
    logging.info("inicio de entrenamiento | epochs=%d | steps/epoch=%d | total_steps=%d",
                 int(getValue(cfg, "epochs")), steps_per_ep, total_steps)

    ctx = make_sdpa_ctx() if device == "cuda" else contextlib.nullcontext()
    with ctx:
        for ep in range(1, int(getValue(cfg, "epochs"))+1):
            model.train()
            opt.zero_grad(set_to_none=True)
            grad_accum = max(1, int(getValue(cfg, "grad_accum")))
            ep_loss_sum, steps_seen = 0.0, 0
            bs = int(getValue(cfg, "batch_size"))
            pbar = tqdm(dl_tr, total=len(dl_tr), desc=f"train e{ep}", dynamic_ncols=True)
            t0 = time.time()

            for step, (encA, encB, y, _) in enumerate(pbar, start=1):
                encA = {k: v.to(device, non_blocking=True) for k, v in encA.items()}
                encB = {k: v.to(device, non_blocking=True) for k, v in encB.items()}
                y = y.to(device, non_blocking=True)
                try:
                    with autocast(**amp_kwargs):
                        out = model(encA, encB)
                        loss = loss_fn(out, {"label": y}) / grad_accum
                    if not torch.isfinite(loss):
                        logging.warning("loss no finita en step=%d, salto", step)
                        opt.zero_grad(set_to_none=True)
                        if device=="cuda": torch.cuda.empty_cache()
                        continue
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device=="cuda":
                        logging.error("oom en step=%d, limpiando", step)
                        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                        continue
                    raise

                scaler.scale(loss).backward()
                stepped = False
                if step % grad_accum == 0:
                    clip_norm = float(getValue(cfg, "clip_norm"))
                    if clip_norm and clip_norm > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sch.step()
                    stepped = True

                loss_item = float(loss.item() * grad_accum)
                ep_loss_sum += loss_item; steps_seen += 1

                if step % step_interval == 0 or step == 1:
                    pd.DataFrame([{
                        "time": nowStr(),
                        "epoch": ep, "step": step, "loss": round(loss_item, 6),
                        "lr": sch.get_last_lr()[0] if stepped else getLr(opt)
                    }]).to_csv(step_csv, mode="a", header=False, index=False)

                    elapsed = time.time() - t0
                    seen_samples = steps_seen * bs
                    tps = seen_samples / max(1e-6, elapsed)
                    pbar.set_postfix({
                        "loss": f"{loss_item:.4f}",
                        "lr": f"{(sch.get_last_lr()[0] if stepped else getLr(opt)):.2e}",
                        "tps": f"{tps:.1f}/s",
                        "mem": getGpuMem()
                    })

            mean_train_loss = ep_loss_sum / max(1, steps_seen)
            stats = dict(val_loss=np.nan, val_acc=np.nan, macro_f1=np.nan, entropy=np.nan, dist=[np.nan]*3)
            if dl_va is not None:
                model.eval()
                stats = runValidation(model, dl_va, device, use_amp, ep, {"reports_dir": reports_dir, "cfg": cfg})

            if dl_va is not None and collapseAlert(stats["val_acc"], stats["entropy"], stats["dist"]):
                pd.DataFrame([{
                    "time": nowStr(),
                    "epoch": ep, "type": "collapse_33_33_33",
                    "detail": f"acc={stats['val_acc']:.3f}, ent={stats['entropy']:.3f}, dist={stats['dist']}"
                }]).to_csv(alerts_csv, mode="a", header=False, index=False)
                logging.warning("posible colapso: %s", stats["dist"])

            per = stats.get("per", {"A":{}, "B":{}, "TIE":{}})
            def g(c, k): 
                v = per.get(c, {}).get(k, np.nan)
                return round(float(v), 6) if not np.isnan(v) else np.nan

            pd.DataFrame([{
                "time": nowStr(),
                "epoch": ep,
                "mean_train_loss": round(mean_train_loss, 6),
                "val_loss": round(float(stats["val_loss"]), 6) if not np.isnan(stats["val_loss"]) else np.nan,
                "val_acc": round(float(stats["val_acc"]), 6) if not np.isnan(stats["val_acc"]) else np.nan,
                "macro_f1": round(float(stats["macro_f1"]), 6) if not np.isnan(stats["macro_f1"]) else np.nan,
                "entropy": round(float(stats["entropy"]), 6) if not np.isnan(stats["entropy"]) else np.nan,
                "dist_A": round(float(stats["dist"][0]), 6) if not np.isnan(stats["dist"][0]) else np.nan,
                "dist_B": round(float(stats["dist"][1]), 6) if not np.isnan(stats["dist"][1]) else np.nan,
                "dist_TIE": round(float(stats["dist"][2]), 6) if not np.isnan(stats["dist"][2]) else np.nan,
                "precision_A": g("A","precision"),
                "recall_A":    g("A","recall"),
                "f1_A":        g("A","f1"),
                "precision_B": g("B","precision"),
                "recall_B":    g("B","recall"),
                "f1_B":        g("B","f1"),
                "precision_TIE": g("TIE","precision"),
                "recall_TIE":    g("TIE","recall"),
                "f1_TIE":        g("TIE","f1"),
                "steps": steps_seen
            }]).to_csv(epoch_csv, mode="a", header=False, index=False)

            # guardar modelo "mejor" según la métrica de monitor
            save_key = getValue(cfg, "monitor.save_best_by")
            metric = (-mean_train_loss) if (dl_va is None) else stats.get(save_key, np.nan)
            is_better = (best_metric is None) or (metric > best_metric)
            if is_better:
                best_metric = metric
                torch.save(model.state_dict(), os.path.join(out_dir, "model.bin"))
                try:
                    model.backbone.config.save_pretrained(out_dir)
                except Exception:
                    pass
                AutoTokenizer.from_pretrained(getValue(cfg, "pretrained_name")).save_pretrained(os.path.join(out_dir, "tokenizer"))
                yaml.safe_dump(cfg, open(os.path.join(out_dir, "train_config.yaml"), "w"))

            # early stopping
            es_mode = getValue(cfg, "early_stopping.mode")
            es_pat  = int(getValue(cfg, "early_stopping.patience"))
            es_key  = getValue(cfg, "early_stopping.metric")
            cur = stats.get(es_key, np.nan)
            if best_metric is None or (es_mode=="max" and cur >= best_metric) or (es_mode=="min" and cur <= best_metric):
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= es_pat:
                logging.warning("early stopping por paciencia=%d", es_pat)
                break

    dur = time.time() - start
    logging.info("duración: %.1f s (%.2f h)", dur, dur/3600.0)
    logging.info("modelo guardado en: %s", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)      # ej: data/clean/train_strat.csv
    ap.add_argument("--out", required=True)        # ej: results/deberta/run_xxx
    ap.add_argument("--cfg", default="m_pair-ranker-deberta/configs/default.yaml")
    ap.add_argument("--valid", default=None)       # ej: data/clean/valid_strat.csv
    args = ap.parse_args()
    trainModel(args.train, args.out, args.cfg, args.valid)

