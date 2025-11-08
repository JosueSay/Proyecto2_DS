import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from config_loader import getValue

def filterEnc(enc: dict) -> dict:
    # filtra llaves nulas o ausentes (roberta no usa token_type_ids)
    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }
    if "token_type_ids" in enc and enc["token_type_ids"] is not None:
        out["token_type_ids"] = enc["token_type_ids"]
    return out

class CrossEncoder(nn.Module):
    # cross-encoder que puntúa (prompt, respA) y (prompt, respB) con logits [A, B, TIE]
    def __init__(self, model_name: str, cfg: dict):
        super().__init__()

        try:
            dropout = float(getValue(cfg, "dropout"))
            grad_checkpointing = bool(getValue(cfg, "grad_checkpointing"))
            compile_flag = bool(getValue(cfg, "compile"))
        except KeyError as e:
            raise KeyError(f"falta en default.yaml: {e}. agrega la clave requerida") from e

        # ajusta dropout del backbone desde cfg
        hf_cfg = AutoConfig.from_pretrained(model_name)
        if hasattr(hf_cfg, "hidden_dropout_prob"):
            hf_cfg.hidden_dropout_prob = dropout
        if hasattr(hf_cfg, "attention_probs_dropout_prob"):
            hf_cfg.attention_probs_dropout_prob = dropout

        self.backbone = AutoModel.from_pretrained(model_name, config=hf_cfg)

        # habilita grad checkpointing solo si el yaml lo indica
        if grad_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            if hasattr(self.backbone, "config") and hasattr(self.backbone.config, "use_cache"):
                self.backbone.config.use_cache = False
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.backbone.gradient_checkpointing_enable()

        hid = hf_cfg.hidden_size

        # cabeza de score individual (para monitoreo)
        self.score_head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
        )

        # cabeza multiclase con features combinadas
        comb_in = hid * 4
        self.classifier = nn.Sequential(
            nn.Linear(comb_in, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 3),  # logits [A, B, TIE]
        )

        # compila si está habilitado en yaml
        if compile_flag and hasattr(torch, "compile"):
            self.pooled = torch.compile(self.pooled)
            self.forward = torch.compile(self.forward)

    def pooled(self, enc: dict) -> torch.Tensor:
        # usa representación [cls] (posición 0)
        out = self.backbone(**filterEnc(enc), return_dict=False)
        return out[0][:, 0]

    def score(self, enc: dict) -> torch.Tensor:
        # score escalar por pareja
        h = self.pooled(enc)
        return self.score_head(h).squeeze(-1)

    def forward(self, enc_a: dict, enc_b: dict):
        h_a = self.pooled(enc_a)
        h_b = self.pooled(enc_b)

        h_abs = torch.abs(h_a - h_b)
        h_mul = h_a * h_b
        feats = torch.cat([h_a, h_b, h_abs, h_mul], dim=-1)
        logits = self.classifier(feats)

        # scores con gradiente (para loss.type=bt)
        s_a = self.score_head(h_a).squeeze(-1)
        s_b = self.score_head(h_b).squeeze(-1)
        return logits, s_a, s_b

    @torch.no_grad()
    def predictProbs(self, enc_a: dict, enc_b: dict) -> torch.Tensor:
        # devuelve probabilidades softmax [B, 3]
        self.eval()
        logits, _, _ = self.forward(enc_a, enc_b)
        return torch.softmax(logits, dim=-1)
