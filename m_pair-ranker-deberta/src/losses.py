import torch
import torch.nn.functional as F
from config_loader import getValue

def getTargets(batch):
    # soporta batch como dict {"label": y} o tupla (..., y, ...)
    if isinstance(batch, dict):
        return batch["label"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        return batch[2]
    raise ValueError("batch no contiene 'label'")

def crossEntropyLoss(logits, targets, label_smoothing: float = 0.0, class_weights=None):
    # ce en fp32 con smoothing y pesos opcionales
    if class_weights is not None and class_weights.device != logits.device:
        class_weights = class_weights.to(logits.device)
    return F.cross_entropy(
        logits.float(),
        targets,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
        reduction="mean",
    )

def btLoss(score_a, score_b, y_bin, y_tie=None):
    # bradley-terry/ranknet binario con soporte de empates
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

def makeLoss(cfg):
    # construye closure de pérdida con valores del yaml
    try:
        loss_type = str(getValue(cfg, "loss.type")).lower()
        label_smoothing = float(getValue(cfg, "loss.label_smoothing"))
    except KeyError as e:
        raise KeyError(f"falta en default.yaml: {e}. agrega la clave requerida en 'loss.*'") from e

    try:
        cw = getValue(cfg, "loss.class_weights")
    except KeyError:
        cw = None

    if cw is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
    else:
        class_weights = None

    if loss_type in {"ce", "cross_entropy", "crossentropy"}:
        def lossFn(outputs, batch):
            # outputs: tuple => (logits, s_a, s_b) o dict => {"logits": ...}
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            targets = getTargets(batch).long().to(logits.device)
            return crossEntropyLoss(logits, targets, label_smoothing, class_weights)
        return lossFn

    if loss_type in {"bt", "bradley-terry", "ranknet"}:
        def lossFn(outputs, batch):
            # outputs: (logits, s_a, s_b) o (s_a, s_b)
            if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                _, s_a, s_b = outputs
            else:
                s_a, s_b = outputs
            y = getTargets(batch).to(s_a.device)
            y_bin = (y == 0).float()  # 1 si gana a
            y_tie = (y == 2).float()  # 1 si empate
            loss, _ = btLoss(s_a, s_b, y_bin, y_tie)
            return loss
        return lossFn

    raise ValueError(f"tipo de loss no soportado: {loss_type}. agrega un valor válido en loss.type")
