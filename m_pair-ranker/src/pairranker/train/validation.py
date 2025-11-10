import os
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from pairranker.losses.factory import makeLoss

def perClassMetrics(conf_mat):
    out, names = {}, ["A","B","TIE"]
    for c, name in enumerate(names):
        tp = int(conf_mat[c, c])
        fp = int(conf_mat[:, c].sum() - tp)
        fn = int(conf_mat[c, :].sum() - tp)
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
    cfg = logs["cfg"]
    loss_fn = makeLoss(cfg)

    loss_type = str(cfg["loss"]["type"]).lower()
    is_bt = loss_type in {"bt", "bradley-terry", "ranknet"}
    ev = cfg.get("eval", {})
    T   = float(ev.get("bt_temp", 1.0))
    tau = float(ev.get("tie_tau", 0.5))
    alpha = float(ev.get("tie_alpha", 0.6))

    amp_kwargs = dict(device_type="cuda", dtype=torch.bfloat16) if (use_amp and device=="cuda") else dict(device_type=device, enabled=False)

    all_probs, all_y, losses, steps = [], [], 0.0, 0
    seq_lens = []
    all_delta, all_sa, all_sb = [], [], []

    pbar = tqdm(loader, total=len(loader), desc=f"val {epoch_idx}", leave=False, dynamic_ncols=True)
    for encA, encB, y, _, _ in pbar:
        encA = {k: v.to(device, non_blocking=True) for k, v in encA.items()}
        encB = {k: v.to(device, non_blocking=True) for k, v in encB.items()}
        y = y.to(device, non_blocking=True)

        with autocast(**amp_kwargs):
            out = model(encA, encB)
            loss = loss_fn(out, {"label": y})

            if is_bt:
                if isinstance(out, (tuple, list)):
                    if len(out) == 3:
                        _, s_a, s_b = out
                    elif len(out) == 2:
                        s_a, s_b = out
                    else:
                        raise ValueError(f"salida inválida para BT: len={len(out)}")
                else:
                    raise ValueError("salida inválida para BT (se esperaba tuple/list)")

                delta = torch.clamp(s_a - s_b, -20.0, 20.0)
                pA = torch.sigmoid(delta / max(1e-6, T))
                pB = 1.0 - pA
                pT = torch.exp(-torch.abs(delta) / max(1e-6, tau)) * float(alpha)
                pT = torch.clamp(pT, 0.0, 0.95)
                probs = torch.stack([pA * (1.0 - pT), pB * (1.0 - pT), pT], dim=-1)

                all_delta.append(delta.detach().cpu().float())
                all_sa.append(s_a.detach().cpu().float())
                all_sb.append(s_b.detach().cpu().float())
            else:
                logits = out[0] if isinstance(out, (tuple, list)) else out
                logits = logits - logits.max(dim=-1, keepdim=True).values
                probs = torch.softmax(logits, dim=-1)

        losses += float(loss.detach().cpu()); steps += 1
        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

        la = encA["attention_mask"].sum(dim=1).detach().cpu().numpy()
        lb = encB["attention_mask"].sum(dim=1).detach().cpu().numpy()
        seq_lens.append(np.maximum(la, lb))

        if steps % 10 == 0:
            pbar.set_postfix(loss=f"{losses/steps:.4f}")

    if steps == 0:
        return dict(val_loss=np.nan, val_acc=np.nan, macro_f1=np.nan, entropy=np.nan, dist=[np.nan]*3, conf=None, per=None)

    probs_np = np.concatenate(all_probs, 0)
    y_np = np.concatenate(all_y, 0)
    seq_np = np.concatenate(seq_lens, 0).astype(int)

    acc, macro_f1, ent, dist, conf = classMetrics(probs_np, y_np)
    per = perClassMetrics(conf)

    rep_dir = logs["reports_dir"]; df_src = loader.dataset.df.reset_index(drop=True)
    y_pred = probs_np.argmax(1).astype(int)

    out_df_dict = {
        "p_A": probs_np[:,0], "p_B": probs_np[:,1], "p_TIE": probs_np[:,2],
        "y_true": y_np.astype(int), "y_pred": y_pred,
        "prompt_len": df_src.get("prompt_len", pd.Series([np.nan]*len(df_src))).iloc[:len(y_np)].to_list(),
        "respA_len":  df_src.get("respA_len",  pd.Series([np.nan]*len(df_src))).iloc[:len(y_np)].to_list(),
        "respB_len":  df_src.get("respB_len",  pd.Series([np.nan]*len(df_src))).iloc[:len(y_np)].to_list(),
        "seq_len_total": seq_np,
    }

    if is_bt:
        delta_np = torch.cat(all_delta, 0).numpy()
        sa_np    = torch.cat(all_sa, 0).numpy()
        sb_np    = torch.cat(all_sb, 0).numpy()
        out_df_dict.update({"delta": delta_np, "s_a": sa_np, "s_b": sb_np})

    out_df = pd.DataFrame(out_df_dict)
    if "id" in df_src.columns:
        out_df.insert(0, "id", df_src["id"].iloc[:len(y_np)].to_list())
    out_df.to_csv(os.path.join(rep_dir, f"val_predictions_epoch_{epoch_idx:02d}.csv"), index=False)

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
    path_pd = os.path.join(rep_dir, "pred_distributions.csv")
    pd.DataFrame(dist_rows).to_csv(path_pd, mode="a", header=not os.path.exists(path_pd), index=False)

    pd.DataFrame(conf, columns=["pred_A","pred_B","pred_TIE"]).assign(true=["A","B","TIE"])\
      .to_csv(os.path.join(rep_dir, f"confusion_epoch_{epoch_idx:02d}.csv"), index=False)
    pd.DataFrame([
        {"class":"A", **per["A"]},
        {"class":"B", **per["B"]},
        {"class":"TIE", **per["TIE"]},
    ]).to_csv(os.path.join(rep_dir, f"class_report_epoch_{epoch_idx:02d}.csv"), index=False)

    pd.DataFrame(conf, columns=["pred_A","pred_B","pred_TIE"]).assign(true=["A","B","TIE"]).to_csv(os.path.join(rep_dir, "confusion.csv"), index=False)
    pd.DataFrame([
        {"class":"A", **per["A"]},
        {"class":"B", **per["B"]},
        {"class":"TIE", **per["TIE"]},
    ]).to_csv(os.path.join(rep_dir, "class_report.csv"), index=False)

    return dict(val_loss=losses/steps, val_acc=acc, macro_f1=macro_f1, entropy=ent, dist=dist, conf=conf, per=per)
