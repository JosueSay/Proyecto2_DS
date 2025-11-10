import math
import torch
import torch.nn.functional as F
from pairranker.config.loader import getValue

def getTargets(batch):
    # acepta dict {"label": y} o tupla (encA, encB, y, ...)
    if isinstance(batch, dict):
        return batch["label"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        return batch[2]
    raise ValueError("batch no contiene 'label'")

def btLoss(score_a, score_b, y_bin, y_tie=None):
    # y_bin: 1 si gana A, 0 si gana B
    # y_tie: 1 si empate, 0 en otro caso
    delta = score_a - score_b
    p_a = torch.sigmoid(delta).clamp(1e-7, 1 - 1e-7)
    if y_tie is None:
        return F.binary_cross_entropy(p_a, y_bin), p_a
    z = torch.where(y_tie > 0.5, torch.full_like(y_bin, 0.5), y_bin)
    loss = - (z * torch.log(p_a) + (1 - z) * torch.log(1 - p_a)).mean()
    return loss, p_a

def buildBradleyTerryLoss(cfg: dict):
    def lossFn(outputs, batch):
        # outputs: (logits, s_a, s_b) o (s_a, s_b)
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 3:
                _, s_a, s_b = outputs
            elif len(outputs) == 2:
                s_a, s_b = outputs
            else:
                raise ValueError(f"salida no válida para BT: len={len(outputs)}")
        else:
            raise ValueError("salida no válida para BT (se esperaba tuple/list)")

        y = getTargets(batch).to(s_a.device)   # 0=A, 1=B, 2=TIE
        y_bin = (y == 0).float()
        y_tie = (y == 2).float()
        loss, _ = btLoss(s_a, s_b, y_bin, y_tie)
        return loss

    # info para epochs.csv (deja NaN en campos CE)
    lossFn.log_info = {
        "loss_type": "bt",
        "label_smoothing": float('nan'),
        "w_A": float('nan'),
        "w_B": float('nan'),
        "w_TIE": float('nan'),
    }
    return lossFn