from pairranker.config.loader import getValue
from .cross_entropy import buildCrossEntropyLoss
from .bradley_terry import buildBradleyTerryLoss

# valores válidos: {"cross_entropy", "bradley_terry"}
def makeLoss(cfg: dict):
    loss_type = str(getValue(cfg, "loss.type")).lower()
    if loss_type == "cross_entropy":
        return buildCrossEntropyLoss(cfg)
    if loss_type == "bradley_terry":
        return buildBradleyTerryLoss(cfg)
    raise ValueError(f"loss.type no soportado: {loss_type}. usa 'cross_entropy' o 'bradley_terry'")
