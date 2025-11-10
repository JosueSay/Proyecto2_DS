import logging
from transformers import AutoConfig, AutoModel

def buildBackbone(pretrained_name: str, dropout: float, grad_checkpointing: bool):
    cfg = AutoConfig.from_pretrained(pretrained_name)
    # aplica dropout si existe en el config
    if hasattr(cfg, "hidden_dropout_prob"):
        cfg.hidden_dropout_prob = dropout
    if hasattr(cfg, "attention_probs_dropout_prob"):
        cfg.attention_probs_dropout_prob = dropout

    model = AutoModel.from_pretrained(pretrained_name, config=cfg)

    # intenta activar gc solo si procede
    if grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        # algunos modelos exponen el método pero NO lo soportan (XLNet lanza ValueError)
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except Exception as e:
            logging.warning("gradient checkpointing no soportado por %s: %s",
                            model.__class__.__name__, e)

    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    if hidden_size is None:
        raise ValueError("no se pudo determinar hidden_size del backbone")
    return model, hidden_size
