import os, math, time, yaml, argparse, logging, random, contextlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm.auto import tqdm

from data import PairDataset, collateFn
from model import CrossEncoder
from losses import makeLoss
from config_loader import loadYamlConfig, getValue

# ================== utils ==================
def setSeed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensureDirs(path): os.makedirs(path, exist_ok=True)
def nowStr(): return time.strftime("%Y-%m-%d %H:%M:%S")
def getLr(optimizer): return optimizer.param_groups[0]["lr"]

def gpuMemMB():
    if torch.cuda.is_available():
        cur = torch.cuda.memory_allocated() / (1024**2)
        return float(cur)
    return 0.0

def gradGlobalNorm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.data.norm(2)
            total += float(n.item()**2)
    return math.sqrt(total) if total>0 else 0.0

def make_sdpa_ctx():
    import contextlib
    return contextlib.nullcontext()

# ============== métricas / validación ==============
def perClassMetrics(conf_mat):
    out, names = {}, ["A","B","TIE"]
    for c, name in enumerate(names):
      tp = int(conf_mat[c, c]); fp = int(conf_mat[:, c].sum() - tp); fn = int(conf_mat[c, :].sum() - tp)
      prec = tp / (tp + fp) if (tp + fp) else 0.0
      rec  = tp / (tp + fn) if (tp + fn) else 0.0
      f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
      out[name] = dict(precision=prec, recall=rec, f1=f1)
    return out

def classMetrics(probs_np, y_np):
    y_pred = probs_np.argmax(1)
    acc = float((y_pred == y_np).mean())
    mat = np.zeros((3,3), dtype=int)
    for t, p in zip(y_np, y_pred): mat[t, p] += 1
    f1s = []
    for c in range(3):
        tp = int(((y_pred == c) & (y_np == c)).sum())
        fp = int(((y_pred == c) & (y_np != c)).sum())
        fn = int(((y_pred != c) & (y_np == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    ent = float((-np.sum(probs_np * np.log(probs_np + 1e-8), axis=1)).mean())
    dist = probs_np.mean(0).tolist()
    return acc, macro_f1, ent, dist, mat

def collapseAlert(dist, val_acc, ent, tol=0.05):
    uniform = all(abs(d - 1/3) <= tol for d in dist)
    return uniform and (val_acc < 0.36) and (ent > 0.95)

@torch.inference_mode()
def runValidation(model, loader, device, use_amp, epoch_idx, logs):
    loss_fn = makeLoss(logs["cfg"])
    amp_kwargs = dict(device_type="cuda", dtype=torch.bfloat16) if (use_amp and device=="cuda") else dict(device_type=device, enabled=False)

    all_probs, all_y, losses, steps = [], [], 0.0, 0
    seq_lens = []  # para val_predictions
    idx0 = 0

    pbar = tqdm(loader, total=len(loader), desc=f"val {epoch_idx}", leave=False, dynamic_ncols=True)
    for encA, encB, y, _, _ in pbar:
        bs = y.size(0)
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

        # seq_len_total por fila (max entre A/B)
        la = encA["attention_mask"].sum(dim=1).detach().cpu().numpy()
        lb = encB["attention_mask"].sum(dim=1).detach().cpu().numpy()
        seq_lens.append(np.maximum(la, lb))

        if steps % 10 == 0:
            pbar.set_postfix(loss=f"{losses/steps:.4f}")
        idx0 += bs

    if steps == 0:
        return dict(val_loss=np.nan, val_acc=np.nan, macro_f1=np.nan, entropy=np.nan, dist=[np.nan]*3, conf=None, per=None)

    probs_np = np.concatenate(all_probs, 0)
    y_np = np.concatenate(all_y, 0)
    seq_np = np.concatenate(seq_lens, 0).astype(int)

    acc, macro_f1, ent, dist, conf = classMetrics(probs_np, y_np)
    per = perClassMetrics(conf)

    # -------- outputs de valid ----------
    rep_dir = logs["reports_dir"]; df_src = loader.dataset.df.reset_index(drop=True)
    y_pred = probs_np.argmax(1).astype(int)

    # predicciones de validación (con longitudes del CSV y seq_len_total)
    out_df = pd.DataFrame({
        "p_A": probs_np[:,0], "p_B": probs_np[:,1], "p_TIE": probs_np[:,2],
        "y_true": y_np.astype(int), "y_pred": y_pred,
        "prompt_len": df_src.get("prompt_len", pd.Series([np.nan]*len(df_src))).iloc[:len(y_np)].to_list(),
        "respA_len":  df_src.get("respA_len",  pd.Series([np.nan]*len(df_src))).iloc[:len(y_np)].to_list(),
        "respB_len":  df_src.get("respB_len",  pd.Series([np.nan]*len(df_src))).iloc[:len(y_np)].to_list(),
        "seq_len_total": seq_np,
    })
    if "id" in df_src.columns:
        out_df.insert(0, "id", df_src["id"].iloc[:len(y_np)].to_list())
    out_df.to_csv(os.path.join(rep_dir, f"val_predictions_epoch_{epoch_idx:02d}.csv"), index=False)

    # distribución por etiqueta (append)
    dist_rows = []
    for cls in [0,1,2]:
        mask = (y_np == cls)
        if mask.any():
            sub = probs_np[mask]
            dist_rows.append({
                "epoch": epoch_idx, "y_true": cls,
                "mean_p_A": float(sub[:,0].mean()), "mean_p_B": float(sub[:,1].mean()), "mean_p_TIE": float(sub[:,2].mean()),
                "var_p_A":  float(sub[:,0].var()),  "var_p_B":  float(sub[:,1].var()),  "var_p_TIE":  float(sub[:,2].var()),
                "n": int(mask.sum())
            })
    pd.DataFrame(dist_rows).to_csv(os.path.join(rep_dir, "pred_distributions.csv"), mode="a", header=not os.path.exists(os.path.join(rep_dir, "pred_distributions.csv")), index=False)

    # confusion + class_report (versionado y último)
    pd.DataFrame(conf, columns=["pred_A","pred_B","pred_TIE"]).assign(true=["A","B","TIE"])\
      .to_csv(os.path.join(rep_dir, f"confusion_epoch_{epoch_idx:02d}.csv"), index=False)
    pd.DataFrame([
        {"class":"A", **per["A"]},
        {"class":"B", **per["B"]},
        {"class":"TIE", **per["TIE"]},
    ]).to_csv(os.path.join(rep_dir, f"class_report_epoch_{epoch_idx:02d}.csv"), index=False)
    # sobrescribe "latest"
    pd.DataFrame(conf, columns=["pred_A","pred_B","pred_TIE"]).assign(true=["A","B","TIE"]).to_csv(os.path.join(rep_dir, "confusion.csv"), index=False)
    pd.DataFrame([
        {"class":"A", **per["A"]},
        {"class":"B", **per["B"]},
        {"class":"TIE", **per["TIE"]},
    ]).to_csv(os.path.join(rep_dir, "class_report.csv"), index=False)

    return dict(val_loss=losses/steps, val_acc=acc, macro_f1=macro_f1, entropy=ent, dist=dist, conf=conf, per=per)

# ================== entrenamiento ==================
def trainModel(train_csv, out_dir, cfg_path="m_pair-ranker-deberta/configs/default.yaml", valid_csv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ensureDirs(out_dir)

    cfg = loadYamlConfig(cfg_path)
    setSeed(getValue(cfg, "seed"))

    reports_dir = getValue(cfg, "logging.reports_dir"); ensureDirs(reports_dir)
    # dump config efectiva al inicio
    run_cfg_path = os.path.join(reports_dir, getValue(cfg, "logging.run_config_used"))
    yaml.safe_dump(cfg, open(run_cfg_path, "w"), sort_keys=False)

    # csvs base
    step_csv   = os.path.join(reports_dir, getValue(cfg, "logging.step_csv"))
    epoch_csv  = os.path.join(reports_dir, getValue(cfg, "logging.epoch_csv"))
    alerts_csv = os.path.join(reports_dir, getValue(cfg, "logging.alerts_csv"))
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("gpu: %s | cuda %s", torch.cuda.get_device_name(0), torch.version.cuda)

    # data
    df_tr = pd.read_csv(train_csv); ds_tr = PairDataset(df_tr, cfg)
    nw  = int(getValue(cfg, "num_workers"))
    pf_tr = int(getValue(cfg, "dataloader.prefetch_factor_train"))
    pf_va = int(getValue(cfg, "dataloader.prefetch_factor_val"))
    pw  = bool(getValue(cfg, "dataloader.persistent_workers"))
    pin = bool(getValue(cfg, "data.pin_memory"))
    bs  = int(getValue(cfg, "batch_size"))

    dl_kwargs_tr = dict(
        dataset=ds_tr,
        batch_size=bs,
        shuffle=bool(getValue(cfg, "data.shuffle")),
        collate_fn=collateFn,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(pw and nw > 0),
    )
    if nw > 0:
        dl_kwargs_tr["prefetch_factor"] = pf_tr
    dl_tr = DataLoader(**dl_kwargs_tr)

    dl_va = None
    if valid_csv:
        df_va = pd.read_csv(valid_csv); ds_va = PairDataset(df_va, cfg)
        nw_va = max(0, nw // 2)
        dl_kwargs_va = dict(
            dataset=ds_va,
            batch_size=int(getValue(cfg, "data.val_batch_size")),
            shuffle=False,
            collate_fn=collateFn,
            num_workers=nw_va,
            pin_memory=pin,
            persistent_workers=(pw and nw_va > 0),
        )
        if nw_va > 0:
            dl_kwargs_va["prefetch_factor"] = pf_va
        dl_va = DataLoader(**dl_kwargs_va)


    model = CrossEncoder(getValue(cfg, "pretrained_name"), cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(getValue(cfg, "lr")), weight_decay=float(getValue(cfg, "weight_decay")))
    steps_per_ep = math.ceil(len(dl_tr.dataset)/bs)
    total_steps  = max(1, int(getValue(cfg, "epochs")) * steps_per_ep // max(1, int(getValue(cfg, "grad_accum"))))
    warmup_steps = int(float(getValue(cfg, "warmup_ratio")) * total_steps)
    sch = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps) if str(getValue(cfg, "scheduler")).lower()=="cosine" \
          else get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    amp_mode = str(getValue(cfg, "amp")).lower()
    use_amp  = (device=="cuda") and (amp_mode != "false")
    amp_kwargs = dict(device_type="cuda", dtype=(torch.bfloat16 if amp_mode=="bf16" else torch.float16)) if use_amp else dict(device_type=device, enabled=False)
    scaler = GradScaler(device if device=="cuda" else "cpu", enabled=(use_amp and amp_mode=="fp16"))

    loss_fn = makeLoss(cfg)
    best_metric, bad_epochs, valloss_bad_epochs = None, 0, 0
    global_step = 0

    # token budget file por época
    budget_tpl = os.path.join(reports_dir, "token_budget_epoch_{:02d}.csv")
    maxp, maxr = int(getValue(cfg,"max_len_prompt")), int(getValue(cfg,"max_len_resp"))
    total_budget_cfg = int(maxp + maxr + 3)

    ctx = make_sdpa_ctx() if device=="cuda" else contextlib.nullcontext()
    with ctx:
        for ep in range(1, int(getValue(cfg, "epochs"))+1):
            # archivo de presupuesto por época
            budget_rows = []

            model.train(); opt.zero_grad(set_to_none=True)
            grad_accum = max(1, int(getValue(cfg, "grad_accum")))
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

                scaler.scale(loss).backward()

                stepped = False
                grad_norm_val = None
                if step % grad_accum == 0:
                    scaler.unscale_(opt)
                    clip_norm = float(getValue(cfg, "clip_norm"))
                    if clip_norm and clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    grad_norm_val = gradGlobalNorm(model)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True); sch.step()
                    stepped = True; global_step += 1

                loss_item = float(loss.item() * grad_accum)
                ep_loss_sum += loss_item; steps_seen += 1

                # logging de steps.csv
                if step == 1 or step % int(getValue(cfg, "logging.step_interval")) == 0:
                    pd.DataFrame([{
                        "time": nowStr(), "epoch": ep, "step": step, "global_step": global_step,
                        "lr": sch.get_last_lr()[0] if stepped else getLr(opt),
                        "loss": round(loss_item, 6),
                        "grad_norm": (round(grad_norm_val, 6) if grad_norm_val is not None else np.nan),
                        "gpu_mem_mb": round(gpuMemMB(), 1),
                    }]).to_csv(step_csv, mode="a", header=False, index=False)

                # medir presupuesto del batch
                la = encA["attention_mask"].sum(dim=1).detach().cpu().numpy()
                lb = encB["attention_mask"].sum(dim=1).detach().cpu().numpy()
                tot = np.maximum(la, lb)
                budget_rows.append({
                    "epoch": ep, "phase": "train", "batch_id": step,
                    "max_len_prompt_cfg": maxp, "max_len_resp_cfg": maxr, "total_budget_cfg": total_budget_cfg,
                    "input_ids_len_max": int(tot.max()), "input_ids_len_mean": float(tot.mean()),
                    "truncated_batch": int((tot >= total_budget_cfg).any())
                })

                # barra
                elapsed = time.time() - t0
                tps = (steps_seen * bs) / max(1e-6, elapsed)
                pbar.set_postfix({"loss": f"{loss_item:.4f}", "lr": f"{(sch.get_last_lr()[0] if stepped else getLr(opt)):.2e}", "tps": f"{tps:.1f}", "mem": f"{gpuMemMB():.0f}MB"})

            # fin epoch train
            pd.DataFrame(budget_rows).to_csv(budget_tpl.format(ep), mode="a", header=not os.path.exists(budget_tpl.format(ep)), index=False)
            mean_train_loss = ep_loss_sum / max(1, steps_seen)

            stats = dict(val_loss=np.nan, val_acc=np.nan, macro_f1=np.nan, entropy=np.nan, dist=[np.nan]*3, per=None)
            if dl_va is not None:
                model.eval()
                # presupuesto valid
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
                # re-iterar para validar (sin gastar dataloader)
                stats = runValidation(model, dl_va, device, use_amp, ep, {"reports_dir": reports_dir, "cfg": cfg})
                pd.DataFrame(budget_rows_val).to_csv(budget_tpl.format(ep), mode="a", header=not os.path.exists(budget_tpl.format(ep)), index=False)

            per = stats.get("per", {"A":{}, "B":{}, "TIE":{}})
            def g(c,k,default=np.nan):
                v = per.get(c, {}).get(k, default)
                return round(float(v), 6) if v==v else np.nan

            # info de loss (label_smoothing y pesos)
            ls_info = getattr(loss_fn, "log_info", {"label_smoothing": np.nan, "w_A": np.nan, "w_B": np.nan, "w_TIE": np.nan})

            # epochs.csv
            pd.DataFrame([{
                "time": nowStr(), "epoch": ep,
                "mean_train_loss": round(mean_train_loss, 6),
                "val_loss": round(float(stats["val_loss"]), 6) if stats["val_loss"]==stats["val_loss"] else np.nan,
                "val_acc":  round(float(stats["val_acc"]), 6)  if stats["val_acc"]==stats["val_acc"]   else np.nan,
                "macro_f1": round(float(stats["macro_f1"]), 6) if stats["macro_f1"]==stats["macro_f1"] else np.nan,
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

            # alertas
            if dl_va is not None:
                # collapse_33_33_33
                if collapseAlert(stats["dist"], stats["val_acc"], stats["entropy"]):
                    pd.DataFrame([{"time": nowStr(), "epoch": ep, "type": "collapse_33_33_33", "value": str(stats["dist"]), "note": "preds ~uniformes"}]).to_csv(alerts_csv, mode="a", header=False, index=False)
                # all_TIE_pred
                y_pred_dist = stats["dist"][2]
                if y_pred_dist >= 0.95:
                    pd.DataFrame([{"time": nowStr(), "epoch": ep, "type": "all_TIE_pred", "value": round(y_pred_dist,4), "note": ">=95% TIE"}]).to_csv(alerts_csv, mode="a", header=False, index=False)
                # flat_probs
                # usando pred_distributions.csv sería más exacto; aquí aproximamos con entropía alta
                if stats["entropy"] >= 1.05:
                    pd.DataFrame([{"time": nowStr(), "epoch": ep, "type": "flat_probs", "value": round(stats['entropy'],4), "note": "alta entropía"}]).to_csv(alerts_csv, mode="a", header=False, index=False)
                # no_progress_loss (2 épocas sin mejorar val_loss)
                cur_vl = stats["val_loss"]
                if not hasattr(trainModel, "_best_vl"): trainModel._best_vl = None
                if trainModel._best_vl is None or (cur_vl < trainModel._best_vl - 1e-6):
                    trainModel._best_vl = cur_vl; valloss_bad_epochs = 0
                else:
                    valloss_bad_epochs += 1
                    if valloss_bad_epochs >= 2:
                        pd.DataFrame([{"time": nowStr(), "epoch": ep, "type": "no_progress_loss", "value": round(float(cur_vl),6), "note": ">=2 épocas sin mejora"}]).to_csv(alerts_csv, mode="a", header=False, index=False)

            # guardar mejor por métrica de monitor
            save_key = str(getValue(cfg, "monitor.save_best_by"))
            metric = (-mean_train_loss) if (dl_va is None) else float(stats.get(save_key, np.nan))
            is_better = (best_metric is None) or (metric > best_metric)
            if is_better:
                best_metric = metric
                torch.save(model.state_dict(), os.path.join(out_dir, "model.bin"))
                try: model.backbone.config.save_pretrained(out_dir)
                except Exception: pass
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

    logging.info("fin del entrenamiento | modelo en: %s", out_dir)

# ================== cli ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="m_pair-ranker-deberta/configs/default.yaml")
    ap.add_argument("--valid", default=None)
    args = ap.parse_args()
    trainModel(args.train, args.out, args.cfg, args.valid)
