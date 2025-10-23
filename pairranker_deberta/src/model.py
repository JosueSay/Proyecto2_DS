import os
import torch, torch.nn as nn
from transformers import AutoModel, AutoConfig

class CrossEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        drop = float(os.getenv("MODEL_DROPOUT", "0.1"))

        cfg = AutoConfig.from_pretrained(model_name)
        # Asegura claves estándar si existen en el backbone
        if hasattr(cfg, "hidden_dropout_prob"):
            cfg.hidden_dropout_prob = drop
        if hasattr(cfg, "attention_probs_dropout_prob"):
            cfg.attention_probs_dropout_prob = drop

        # Carga con la config modificada
        self.backbone = AutoModel.from_pretrained(model_name, config=cfg)

        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        hid = cfg.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Linear(hid, 1)
        )

        # self = torch.compile(self)

    def score(self, enc):
        # enc: dict con input_ids, attention_mask (y token_type_ids si aplica)
        out = self.backbone(**enc)
        # RoBERTa no usa pooler por defecto; tomamos el [CLS]
        pooled = out.last_hidden_state[:, 0]
        return self.head(pooled).squeeze(-1)  # escalar [B]

    def forward(self, enc_a, enc_b):
        return self.score(enc_a), self.score(enc_b)
