import torch
import torch.nn as nn

class ScoreHead(nn.Module):
    # produce score escalar por representación pooled (para BT)
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)

class PairClassifier(nn.Module):
    # combina h_a y h_b y devuelve logits [A, B, TIE]
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        comb_in = hidden_size * 4  # [h_a, h_b, |h_a-h_b|, h_a*h_b]
        self.net = nn.Sequential(
            nn.Linear(comb_in, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, h_a, h_b):
        h_abs = (h_a - h_b).abs()
        h_mul = h_a * h_b
        feats = torch.cat([h_a, h_b, h_abs, h_mul], dim=-1)
        return self.net(feats)
