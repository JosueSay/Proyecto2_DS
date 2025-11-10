import torch
import torch.nn.functional as F
from pairranker.config.loader import getValue

def crossEntropyLoss(logits, targets, label_smoothing: float = 0.0, class_weights=None):
    if class_weights is not None and class_weights.device != logits.device:
        class_weights = class_weights.to(logits.device)
    return F.cross_entropy(
        logits.float(),
        targets,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
        reduction="mean",
    )

def buildCrossEntropyLoss(cfg: dict):
    # parámetros del yaml
    label_smoothing = float(getValue(cfg, "loss.label_smoothing"))
    cw = getValue(cfg, "loss.class_weights")  # [w_A, w_B, w_TIE]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device) if cw is not None else None

    def lossFn(outputs, batch):
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        targets = batch[2].long().to(logits.device)  # (encA, encB, y, ...)
        return crossEntropyLoss(logits, targets, label_smoothing, class_weights)

    # info para epochs.csv
    lossFn.log_info = {
        "loss_type": "cross_entropy",
        "label_smoothing": float(label_smoothing),
        "w_A": float(cw[0]) if cw is not None else 1.0,
        "w_B": float(cw[1]) if cw is not None else 1.0,
        "w_TIE": float(cw[2]) if cw is not None else 1.0,
    }
    return lossFn
