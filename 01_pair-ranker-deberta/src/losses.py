import torch
import torch.nn.functional as F

def btLoss(score_a, score_b, y_bin, y_tie=None):
    """
    Bradley-Terry / RankNet loss estable y compatible con AMP (fp16/bf16).

    Parámetros:
      score_a, score_b : tensores [batch]
      y_bin            : 1 si A gana, 0 si B gana
      y_tie (opcional) : 1 si empate → objetivo = 0.5
    Retorna:
      loss promedio, p_a (probabilidad predicha de que A gane)
    """
    # Diferencia de puntajes
    delta = score_a - score_b

    # Sigmoid estable (usa float32 en AMP para evitar underflow)
    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=torch.is_autocast_enabled()):
        p_a = torch.sigmoid(delta)

        if y_tie is None:
            # RankNet / BT estándar
            loss = F.binary_cross_entropy(p_a, y_bin)
        else:
            # Empates → valor objetivo 0.5
            Z = torch.where(y_tie > 0.5, torch.full_like(y_bin, 0.5), y_bin)
            loss = - (Z * torch.log(p_a.clamp_min(1e-7)) +
                      (1 - Z) * torch.log((1 - p_a).clamp_min(1e-7))).mean()

    return loss, p_a
