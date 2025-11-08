import yaml
from pathlib import Path

REQUIRED_SCHEMA = {
    "model_name": str,
    "pretrained_name": str,
    "max_len_prompt": int,
    "max_len_resp": int,
    "compile": bool,
    "epochs": int,
    "batch_size": int,
    "grad_accum": int,
    "lr": float,
    "weight_decay": float,
    "warmup_ratio": float,
    "scheduler": ("cosine", "linear"),
    "clip_norm": float,
    "seed": int,
    "amp": ("bf16", "fp16", "false"),
    "num_workers": int,
    "dropout": float,
    "grad_checkpointing": bool,
    "early_stopping": {
        "metric": str,
        "mode": ("max", "min"),
        "patience": int,
    },
    "data": {
        "train_csv": str,
        "valid_csv": str,
        "test_csv": str,
        "use_label": bool,
        "use_clean_cols": bool,
        "shuffle": bool,
        "val_batch_size": int,
        "pin_memory": bool,
        "prompt_col": str,
        "respA_col": str,
        "respB_col": str,
    },
    "loss": {
        "type": ("cross_entropy", "bt", "bradley-terry", "ranknet"),
        "label_smoothing": float,
    },
    "logging": {
        "reports_dir": str,
        "runs_dir": str,
        "step_csv": str,
        "epoch_csv": str,
        "alerts_csv": str,
        "confusion_csv": str,
        "class_report_csv": str,
        "preds_sample_csv": str,
        "step_interval": int,
    },
    "monitor": {
        "detect_collapse": bool,
        "save_best_by": str,
        "save_last": bool,
        "verbose": bool,
    },
    "dataloader": {
        "prefetch_factor_train": int,
        "prefetch_factor_val": int,
        "persistent_workers": bool,
    },
    "env": {
        "tokenizers_parallelism": bool,
        "cuda_launch_blocking": int,
        "pytorch_cuda_alloc_conf": str,
        "hf_home": str,
        "use_slow_tokenizer": bool,
    },
}

def loadYamlConfig(path_yaml: str) -> dict:
    # lee yaml, valida estructura y tipos definidos en el esquema
    path = Path(path_yaml)
    if not path.exists():
        raise FileNotFoundError(f"no existe el yaml: {path_yaml}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("el yaml debe mapear a un dict de nivel raíz")
    checkNoExtraKeys(data, REQUIRED_SCHEMA, root="config")
    checkMissingKeys(data, REQUIRED_SCHEMA, root="config")
    return validateAndCoerce(data, REQUIRED_SCHEMA, root="config")

def checkMissingKeys(node: dict, schema: dict | tuple | type, root: str) -> None:
    # asegura que no falten claves esperadas
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise TypeError(f"{root} debe ser dict")
        missing = [k for k in schema.keys() if k not in node]
        if missing:
            raise KeyError(f"faltan claves en {root}: {', '.join(missing)}")
        for k, sub in schema.items():
            checkMissingKeys(node[k], sub, f"{root}.{k}")

def checkNoExtraKeys(node: dict, schema: dict | tuple | type, root: str) -> None:
    # rechaza claves no definidas en el esquema
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise TypeError(f"{root} debe ser dict")
        extra = [k for k in node.keys() if k not in schema]
        if extra:
            raise KeyError(f"claves no reconocidas en {root}: {', '.join(extra)}")
        for k, sub in schema.items():
            if k in node:
                checkNoExtraKeys(node[k], sub, f"{root}.{k}")

def validateAndCoerce(node: dict | str | int | float | bool,
                      schema: dict | tuple | type,
                      root: str):
    # valida tipos y hace coerción segura (por ejemplo str -> float/int)
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise TypeError(f"{root} debe ser dict")
        return {k: validateAndCoerce(node[k], sub, f"{root}.{k}") for k, sub in schema.items()}

    if isinstance(schema, tuple):
        # enum de strings válidos
        if not isinstance(node, str):
            raise TypeError(f"{root} debe ser str y uno de {schema}, recibido {type(node).__name__}")
        if node not in schema:
            raise ValueError(f"{root}='{node}' no está en {schema}")
        return node

    if schema is float:
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, str):
            try:
                return float(node.strip())
            except ValueError:
                raise TypeError(f"{root} debe ser float, recibido str no numérico")
        raise TypeError(f"{root} debe ser float, recibido {type(node).__name__}")

    if schema is int:
        if isinstance(node, int):
            return node
        if isinstance(node, float):
            if node.is_integer():
                return int(node)
            raise TypeError(f"{root} debe ser int, recibido float no entero")
        if isinstance(node, str):
            node_s = node.strip()
            if node_s.isdigit() or (node_s.startswith("-") and node_s[1:].isdigit()):
                return int(node_s)
            raise TypeError(f"{root} debe ser int, recibido str no entero")
        raise TypeError(f"{root} debe ser int, recibido {type(node).__name__}")

    if schema is bool:
        if isinstance(node, bool):
            return node
        raise TypeError(f"{root} debe ser bool, recibido {type(node).__name__}")

    if schema is str:
        if isinstance(node, str):
            return node
        raise TypeError(f"{root} debe ser str, recibido {type(node).__name__}")

    raise TypeError(f"{root} tiene un tipo de esquema no soportado")

def getValue(cfg: dict, path: str):
    # acceso tipo 'data.train_csv' sin fallback
    parts = path.split(".")
    cur = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"ruta inválida o faltante: {path}")
        cur = cur[p]
    return cur
