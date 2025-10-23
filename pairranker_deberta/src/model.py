import torch, torch.nn as nn
from transformers import AutoModel, AutoConfig

class CrossEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hid = cfg.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1)
        )

    def score(self, enc):
        out = self.backbone(**enc)
        pooled = out.last_hidden_state[:, 0]     # [CLS]
        return self.head(pooled).squeeze(-1)     # escalar

    def forward(self, enc_a, enc_b):
        return self.score(enc_a), self.score(enc_b)
