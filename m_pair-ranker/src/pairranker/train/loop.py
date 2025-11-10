import os, math, time, yaml, argparse, logging, contextlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm.auto import tqdm

from pairranker.data_p.dataset import PairDataset
from pairranker.data_p.collate import collateFn
from pairranker.models.cross_encoder import CrossEncoder
from pairranker.losses.factory import makeLoss
from pairranker.config.loader import loadYamlConfig, getValue
from pairranker.train.utils import (
    setSeed, ensureDirs, nowStr, getLr, gpuMemMB, gradGlobalNorm, makeSdpaCtx, setupGpuLogging
)
from pairranker.train.validation import runValidation, collapseAlert

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


def _dlKw(dataset, bs, shuffle, coll, nw, pin, pf, pw):
    kw = dict(dataset=dataset, batch_size=bs, shuffle=shuffle, collate_fn=coll,
              num_workers=nw, pin_memory=pin, persistent_workers=(pw and nw>0))
    if nw > 0:
        kw["prefetch_factor"] = pf
    return kw

def trainModel(cfg_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = loadYamlConfig(cfg_path, attach_run_dirs=True)
    setSeed(int(getValue(cfg, "train.seed")))

    run_dir   = cfg["logging_ext"]["run_dir"]
    reports_dir = cfg["logging_ext"]["report_dir"]
    ensureDirs(run_dir); ensureDirs(reports_dir)

    with open(os.path.join(reports_dir, cfg["logging"]["run_config_used"]), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    step_csv   = os.path.join(reports_dir, cfg["logging"]["step_csv"])
    epoch_csv  = os.path.join(reports_dir, cfg["logging"]["epoch_csv"])
    alerts_csv = os.path.join(reports_dir, cfg["logging"]["alerts_csv"])
    for p, cols in [
        (step_csv,  ["time","epoch","step","global_step","lr","loss","grad_norm","gpu_mem_mb"]),
        (epoch_csv, ["time","epoch","mean_train_loss","val_loss","val_acc","macro_f1",
                     "precision_A","recall_A","f1_A","precision_B","recall_B","f1_B",
                     "precision_TIE","recall_TIE","f1_TIE","dist_A","dist_B","dist_TIE","entropy",
                     "label_smoothing","w_A","w_B","w_TIE"]),
        (alerts_csv,["time","epoch","type","value","note"]),
    ]:
        if not os.path.exists(p): pd.DataFrame(columns=cols).to_csv(p, index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        setupGpuLogging()

    df_tr = pd.read_csv(cfg["data"]["train_csv"])
    ds_tr = PairDataset(df_tr, cfg)
    nw  = int(getValue(cfg, "train.num_workers"))
    pf_tr = int(getValue(cfg, "dataloader.prefetch_factor_train"))
    pf_va = int(getValue(cfg, "dataloader.prefetch_factor_val"))
    pw  = bool(getValue(cfg, "dataloader.persistent_workers"))
    pin = bool(getValue(cfg, "data.pin_memory"))

    bs  = int(getValue(cfg, "train.batch_size"))
    dl_tr = DataLoader(**_dlKw(ds_tr, bs, bool(getValue(cfg, "data.shuffle")), collateFn, nw, pin, pf_tr, pw))

    dl_va = None
    if cfg["data"].get("valid_csv"):
        df_va = pd.read_csv(cfg["data"]["valid_csv"])
        ds_va = PairDataset(df_va, cfg)
        nw_va = max(0, nw//2)
        dl_va = DataLoader(**_dlKw(ds_va, int(getValue(cfg, "data.val_batch_size")), False, collateFn, nw_va, pin, pf_va, pw))

    mcfg = cfg["model"]
    model = CrossEncoder(
        mcfg["pretrained_name"],
        float(mcfg["dropout"]),
        bool(mcfg["grad_checkpointing"]),
        bool(mcfg["compile"])
    ).to(device)    
    
    opt = torch.optim.AdamW(model.parameters(), lr=float(getValue(cfg, "train.lr")),
                            weight_decay=float(getValue(cfg, "train.weight_decay")))
    steps_per_ep = math.ceil(len(dl_tr.dataset) / bs)
    total_steps  = max(1, int(getValue(cfg, "train.epochs")) * steps_per_ep // max(1, int(getValue(cfg, "train.grad_accum"))))
    warmup_steps = int(float(getValue(cfg, "train.warmup_ratio")) * total_steps)

    scheduler_name = str(getValue(cfg, "train.scheduler")).lower()
    sch = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps) if scheduler_name=="cosine" \
          else get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    amp_mode = str(getValue(cfg, "train.amp")).lower()
    use_amp  = (device=="cuda") and (amp_mode != "false")
    amp_kwargs = dict(device_type="cuda", dtype=(torch.bfloat16 if amp_mode=="bf16" else torch.float16)) if use_amp else dict(device_type=device, enabled=False)
    scaler = GradScaler(device if device=="cuda" else "cpu", enabled=(use_amp and amp_mode=="fp16"))

    loss_fn = makeLoss(cfg)
    best_metric, bad_epochs, valloss_bad_epochs = None, 0, 0
    global_step = 0

    budget_tpl = os.path.join(reports_dir, cfg["logging"]["token_budget_tpl"])
    maxp = int(getValue(cfg, "lengths.max_len_prompt"))
    maxr = int(getValue(cfg, "lengths.max_len_resp"))
    total_budget_cfg = int(maxp + maxr + 3)

    ctx = makeSdpaCtx() if device=="cuda" else contextlib.nullcontext()
    with ctx:
        for ep in range(1, int(getValue(cfg, "train.epochs")) + 1):
            budget_rows = []
            model.train(); opt.zero_grad(set_to_none=True)
            grad_accum = max(1, int(getValue(cfg, "train.grad_accum")))
            ep_loss_sum, steps_seen = 0.0, 0
            pbar = tqdm(dl_tr, total=len(dl_tr), desc=f"train e{ep}", dynamic_ncols=True)
            t0 = time.time()

            for step, batch in enumerate(pbar, start=1):
                encA, encB, y, _, _ = batch
                encA = {k: v.to(device, non_blocking=True) for k, v in encA.items()}
                encB = {k: v.to(device, non_blocking=True) for k, v in encB.items()}
                y = y.to(device, non_blocking=True)

                with autocast(**amp_kwargs):
                    out = model(encA, encB)
                    loss = loss_fn(out, {"label": y}) / grad_accum

                if not torch.isfinite(loss):
                    pd.DataFrame([{
                        "time": nowStr(), "epoch": ep, "step": step, "global_step": global_step,
                        "lr": getLr(opt), "loss": float('nan'), "grad_norm": float('nan'),
                        "gpu_mem_mb": round(gpuMemMB(), 1),
                    }]).to_csv(step_csv, mode="a", header=False, index=False)
                    opt.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()

                stepped = False
                grad_norm_val = None
                if step % grad_accum == 0:
                    scaler.unscale_(opt)
                    for p in model.parameters():
                        if p.grad is not None:
                            torch.nan_to_num_(p.grad, nan=0.0, posinf=1e4, neginf=-1e4)

                    clip_norm = float(getValue(cfg, "train.clip_norm"))
                    if clip_norm and clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    grad_norm_val = gradGlobalNorm(model)

                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True); sch.step()
                    stepped = True; global_step += 1

                loss_item = float(loss.item() * grad_accum)
                ep_loss_sum += loss_item; steps_seen += 1

                if step == 1 or step % int(getValue(cfg, "logging.step_interval")) == 0:
                    pd.DataFrame([{
                        "time": nowStr(), "epoch": ep, "step": step, "global_step": global_step,
                        "lr": sch.get_last_lr()[0] if stepped else getLr(opt),
                        "loss": round(loss_item, 6),
                        "grad_norm": (round(grad_norm_val, 6) if grad_norm_val is not None else np.nan),
                        "gpu_mem_mb": round(gpuMemMB(), 1),
                    }]).to_csv(step_csv, mode="a", header=False, index=False)

                la = encA["attention_mask"].sum(dim=1).detach().cpu().numpy()
                lb = encB["attention_mask"].sum(dim=1).detach().cpu().numpy()
                tot = np.maximum(la, lb)
                budget_rows.append({
                    "epoch": ep, "phase": "train", "batch_id": step,
                    "max_len_prompt_cfg": maxp, "max_len_resp_cfg": maxr, "total_budget_cfg": total_budget_cfg,
                    "input_ids_len_max": int(tot.max()), "input_ids_len_mean": float(tot.mean()),
                    "truncated_batch": int((tot >= total_budget_cfg).any())
                })

                elapsed = time.time() - t0
                tps = (steps_seen * bs) / max(1e-6, elapsed)
                pbar.set_postfix({"loss": f"{loss_item:.4f}", "lr": f"{(sch.get_last_lr()[0] if stepped else getLr(opt)):.2e}", "tps": f"{tps:.1f}", "mem": f"{gpuMemMB():.0f}MB"})

            pd.DataFrame(budget_rows).to_csv(budget_tpl.format(ep), mode="a", header=not os.path.exists(budget_tpl.format(ep)), index=False)
            mean_train_loss = ep_loss_sum / max(1, steps_seen)

            stats = dict(val_loss=np.nan, val_acc=np.nan, macro_f1=np.nan, entropy=np.nan, dist=[np.nan]*3, per=None)
            if dl_va is not None:
                model.eval()
                budget_rows_val = []
                for b_id, (encA, encB, y, _, _) in enumerate(dl_va, start=1):
                    la = encA["attention_mask"].sum(dim=1).numpy()
                    lb = encB["attention_mask"].sum(dim=1).numpy()
                    tot = np.maximum(la, lb)
                    budget_rows_val.append({
                        "epoch": ep, "phase": "valid", "batch_id": b_id,
                        "max_len_prompt_cfg": maxp, "max_len_resp_cfg": maxr, "total_budget_cfg": total_budget_cfg,
                        "input_ids_len_max": int(tot.max()), "input_ids_len_mean": float(tot.mean()),
                        "truncated_batch": int((tot >= total_budget_cfg).any())
                    })
                from pairranker.train.validation import runValidation  # evitar ciclo de import
                stats = runValidation(model, dl_va, device, use_amp, ep, {"reports_dir": reports_dir, "cfg": cfg})
                pd.DataFrame(budget_rows_val).to_csv(budget_tpl.format(ep), mode="a", header=not os.path.exists(budget_tpl.format(ep)), index=False)

            per = stats.get("per", {"A":{}, "B":{}, "TIE":{}})
            def g(c,k,default=np.nan):
                v = per.get(c, {}).get(k, default)
                return round(float(v), 6) if v==v else np.nan

            ls_info = getattr(loss_fn, "log_info", {"label_smoothing": np.nan, "w_A": np.nan, "w_B": np.nan, "w_TIE": np.nan})

            pd.DataFrame([{
                "time": nowStr(), "epoch": ep,
                "mean_train_loss": round(mean_train_loss, 6),
                "val_loss":  (round(float(stats["val_loss"]), 6) if stats["val_loss"]==stats["val_loss"] else np.nan),
                "val_acc":   (round(float(stats["val_acc"]),  6) if stats["val_acc"]==stats["val_acc"]   else np.nan),
                "macro_f1":  (round(float(stats["macro_f1"]), 6) if stats["macro_f1"]==stats["macro_f1"] else np.nan),
                "precision_A": g("A","precision"), "recall_A": g("A","recall"), "f1_A": g("A","f1"),
                "precision_B": g("B","precision"), "recall_B": g("B","recall"), "f1_B": g("B","f1"),
                "precision_TIE": g("TIE","precision"), "recall_TIE": g("TIE","recall"), "f1_TIE": g("TIE","f1"),
                "dist_A": (round(float(stats["dist"][0]),6) if stats["dist"][0]==stats["dist"][0] else np.nan),
                "dist_B": (round(float(stats["dist"][1]),6) if stats["dist"][1]==stats["dist"][1] else np.nan),
                "dist_TIE": (round(float(stats["dist"][2]),6) if stats["dist"][2]==stats["dist"][2] else np.nan),
                "entropy": (round(float(stats["entropy"]),6) if stats["entropy"]==stats["entropy"] else np.nan),
                "label_smoothing": ls_info.get("label_smoothing"),
                "w_A": ls_info.get("w_A"), "w_B": ls_info.get("w_B"), "w_TIE": ls_info.get("w_TIE"),
            }]).to_csv(epoch_csv, mode="a", header=False, index=False)

            if dl_va is not None:
                from pairranker.train.validation import collapseAlert
                if collapseAlert(stats["dist"], stats["val_acc"], stats["entropy"]):
                    pd.DataFrame([{"time": nowStr(), "epoch": ep, "type": "collapse_33_33_33", "value": str(stats["dist"]), "note": "preds ~uniformes"}]).to_csv(alerts_csv, mode="a", header=False, index=False)
                y_pred_dist = stats["dist"][2]
                if y_pred_dist >= 0.95:
                    pd.DataFrame([{"time": nowStr(), "epoch": ep, "type": "all_TIE_pred", "value": round(y_pred_dist,4), "note": ">=95% TIE"}]).to_csv(alerts_csv, mode="a", header=False, index=False)
                if stats["entropy"] >= 1.05:
                    pd.DataFrame([{"time": nowStr(), "epoch": ep, "type": "flat_probs", "value": round(stats['entropy'],4), "note": "alta entropía"}]).to_csv(alerts_csv, mode="a", header=False, index=False)

            save_key = str(getValue(cfg, "monitor.save_best_by"))
            metric = (-mean_train_loss) if (dl_va is None) else float(stats.get(save_key, np.nan))
            is_better = (best_metric is None) or (metric > best_metric)
            if is_better:
                best_metric = metric
                torch.save(model.state_dict(), os.path.join(run_dir, "model.bin"))
                try: model.backbone.config.save_pretrained(run_dir)
                except Exception: pass
                AutoTokenizer.from_pretrained(cfg["model"]["pretrained_name"]).save_pretrained(os.path.join(run_dir, "tokenizer"))
                with open(os.path.join(run_dir, "train_config.yaml"), "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

            es_mode = getValue(cfg, "early_stopping.mode")
            es_pat  = int(getValue(cfg, "early_stopping.patience"))
            es_key  = getValue(cfg, "early_stopping.metric")
            cur = stats.get(es_key, np.nan)
            if best_metric is None or (es_mode=="max" and (cur==cur) and cur >= best_metric) or (es_mode=="min" and (cur==cur) and cur <= best_metric):
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= es_pat:
                logging.warning("early stopping por paciencia=%d", es_pat)
                break

    logging.info("fin del entrenamiento | run_dir: %s | reports: %s", run_dir, reports_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    trainModel(args.cfg)
