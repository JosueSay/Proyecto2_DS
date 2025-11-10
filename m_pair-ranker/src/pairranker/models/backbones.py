import torch
from transformers import AutoConfig, AutoModel

def buildBackbone(pretrained_name: str, dropout: float, grad_checkpointing: bool):
    # ajusta dropout en config si existe
    hf_cfg = AutoConfig.from_pretrained(pretrained_name)
    if hasattr(hf_cfg, "hidden_dropout_prob"):
        hf_cfg.hidden_dropout_prob = float(dropout)
    if hasattr(hf_cfg, "attention_probs_dropout_prob"):
        hf_cfg.attention_probs_dropout_prob = float(dropout)

    model = AutoModel.from_pretrained(pretrained_name, config=hf_cfg)

    # grad checkpointing opcional
    if grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    return model, hf_cfg.hidden_size
