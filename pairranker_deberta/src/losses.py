import torch, torch.nn.functional as F

def btLoss(score_a, score_b, y_bin, y_tie=None):
    """
    Bradley-Terry / RankNet.
    y_bin: 1 si A gana, 0 si B gana
    y_tie: 1 si empate (opcional → Z=0.5)
    """
    delta = score_a - score_b
    p_a = torch.sigmoid(delta)
    if y_tie is None:
        loss = F.binary_cross_entropy(p_a, y_bin)
        return loss, p_a
    Z = torch.where(y_tie > 0.5, torch.full_like(y_bin, 0.5), y_bin)
    loss = - (Z*torch.log(p_a + 1e-7) + (1-Z)*torch.log(1-p_a + 1e-7)).mean()
    return loss, p_a
