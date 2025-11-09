import torch
import torch.nn.functional as F
from config_loader import getValue

# extrae y del batch (dict o tupla)
def getTargets(batch):
    if isinstance(batch, dict):
        return batch["label"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        return batch[2]
    raise ValueError("batch no contiene 'label'")

# cross-entropy con smoothing y pesos
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

# bradley-terry/ranknet opcional (compat)
def btLoss(score_a, score_b, y_bin, y_tie=None):
    dev = "cuda" if score_a.is_cuda else "cpu"
    delta = score_a - score_b
    with torch.autocast(device_type=dev, dtype=torch.float32, enabled=torch.is_autocast_enabled()):
        p_a = torch.sigmoid(delta).clamp(1e-7, 1 - 1e-7)
        if y_tie is None:
            loss = F.binary_cross_entropy(p_a, y_bin)
        else:
            z = torch.where(y_tie > 0.5, torch.full_like(y_bin, 0.5), y_bin)
            loss = - (z * torch.log(p_a) + (1 - z) * torch.log(1 - p_a)).mean()
    return loss, p_a

# construye la pérdida desde el yaml y adjunta info para logging
def makeLoss(cfg):
    # tipo
    try:
        loss_type = str(getValue(cfg, "loss.type")).lower()
    except KeyError as e:
        raise KeyError(f"falta en default.yaml: {e}") from e

    # smoothing (default 0.10)
    try:
        label_smoothing = float(getValue(cfg, "loss.label_smoothing"))
    except KeyError:
        label_smoothing = 0.10

    # pesos (default [1.0, 1.0, 0.85] en orden A,B,TIE)
    try:
        cw = getValue(cfg, "loss.class_weights")
    except KeyError:
        cw = [1.0, 1.0, 0.85]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device) if cw is not None else None

    if loss_type in {"ce", "cross_entropy", "crossentropy"}:
        def lossFn(outputs, batch):
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            targets = getTargets(batch).long().to(logits.device)
            return crossEntropyLoss(logits, targets, label_smoothing, class_weights)
        # info para epochs.csv
        lossFn.log_info = {
            "label_smoothing": float(label_smoothing),
            "w_A": float(cw[0]) if cw is not None else 1.0,
            "w_B": float(cw[1]) if cw is not None else 1.0,
            "w_TIE": float(cw[2]) if cw is not None else 1.0,
            "loss_type": "cross_entropy",
        }
        return lossFn

    if loss_type in {"bt", "bradley-terry", "ranknet"}:
        def lossFn(outputs, batch):
            if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                _, s_a, s_b = outputs
            else:
                s_a, s_b = outputs
            y = getTargets(batch).to(s_a.device)
            y_bin = (y == 0).float()
            y_tie = (y == 2).float()
            loss, _ = btLoss(s_a, s_b, y_bin, y_tie)
            return loss
        lossFn.log_info = {"loss_type": "bt"}
        return lossFn

    raise ValueError(f"tipo de loss no soportado: {loss_type}")
