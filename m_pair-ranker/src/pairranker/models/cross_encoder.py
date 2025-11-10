import torch
import torch.nn as nn

from .backbones import buildBackbone
from .heads import ScoreHead, PairClassifier
from pairranker.config.loader import getValue

def filterEnc(enc: dict) -> dict:
    # filtra llaves no usadas (roberta no usa token_type_ids)
    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }
    if "token_type_ids" in enc and enc["token_type_ids"] is not None:
        out["token_type_ids"] = enc["token_type_ids"]
    return out

class CrossEncoder(nn.Module):
    # logits [A,B,TIE] y scores s_a/s_b para pérdidas tipo bradley_terry
    def __init__(self, pretrained_name: str, dropout: float, grad_checkpointing: bool, compile_flag: bool):
        super().__init__()
        self.backbone, hid = buildBackbone(pretrained_name, dropout, grad_checkpointing)
        self.score_head = ScoreHead(hid, dropout)
        self.classifier = PairClassifier(hid, dropout)

        if compile_flag and hasattr(torch, "compile"):
            self.pooled = torch.compile(self.pooled)
            self.forward = torch.compile(self.forward)

    @staticmethod
    def fromConfig(cfg: dict) -> "CrossEncoder":
        # lee yaml nuevo
        pretrained_name = getValue(cfg, "model.pretrained_name")
        dropout = float(getValue(cfg, "model.dropout"))
        grad_checkpointing = bool(getValue(cfg, "model.grad_checkpointing"))
        compile_flag = bool(getValue(cfg, "model.compile"))
        return CrossEncoder(pretrained_name, dropout, grad_checkpointing, compile_flag)

    def pooled(self, enc: dict) -> torch.Tensor:
        out = self.backbone(**filterEnc(enc), return_dict=False)  # (last_hidden_state, ...)
        return out[0][:, 0]  # [cls]

    def score(self, enc: dict) -> torch.Tensor:
        h = self.pooled(enc)
        return self.score_head(h)

    def forward(self, enc_a: dict, enc_b: dict):
        h_a = self.pooled(enc_a)
        h_b = self.pooled(enc_b)
        logits = self.classifier(h_a, h_b)
        s_a = self.score_head(h_a)
        s_b = self.score_head(h_b)
        return logits, s_a, s_b

    @torch.no_grad()
    def predictProbs(self, enc_a: dict, enc_b: dict) -> torch.Tensor:
        self.eval()
        logits, _, _ = self.forward(enc_a, enc_b)
        return torch.softmax(logits, dim=-1)
