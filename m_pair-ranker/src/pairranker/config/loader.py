import yaml
from pathlib import Path
from typing import Any
from datetime import datetime

from .schema import REQUIRED_SCHEMA
from pairranker.utils.io import ensureDir, writeYaml


def checkMissingKeys(node: Any, schema: Any, root: str) -> None:
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise TypeError(f"{root} debe ser dict")
        missing = [k for k in schema.keys() if k not in node]
        if missing:
            raise KeyError(f"faltan claves en {root}: {', '.join(missing)}")
        for k, sub in schema.items():
            checkMissingKeys(node[k], sub, f"{root}.{k}")


def checkNoExtraKeys(node: Any, schema: Any, root: str) -> None:
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise TypeError(f"{root} debe ser dict")
        extra = [k for k in node.keys() if k not in schema]
        if extra:
            raise KeyError(f"claves no reconocidas en {root}: {', '.join(extra)}")
        for k, sub in schema.items():
            if k in node:
                checkNoExtraKeys(node[k], sub, f"{root}.{k}")


def validateAndCoerce(node: Any, schema: Any, root: str):
    if isinstance(schema, dict):
        if not isinstance(node, dict):
            raise TypeError(f"{root} debe ser dict")
        return {k: validateAndCoerce(node[k], sub, f"{root}.{k}") for k, sub in schema.items()}

    if isinstance(schema, tuple):
        if not isinstance(node, str):
            raise TypeError(f"{root} debe ser str y uno de {schema}, recibido {type(node).__name__}")
        if node not in schema:
            raise ValueError(f"{root}='{node}' no está en {schema}")
        return node

    if schema is float:
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, str):
            s = node.strip()
            try:
                return float(s)
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
            s = node.strip()
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                return int(s)
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

    if schema is list:
        if isinstance(node, list):
            return node
        raise TypeError(f"{root} debe ser list, recibido {type(node).__name__}")

    raise TypeError(f"{root} tiene un tipo de esquema no soportado")


def getValue(cfg: dict, path: str):
    parts = path.split(".")
    cur: Any = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"ruta inválida o faltante: {path}")
        cur = cur[p]
    return cur


def deriveRunDirs(cfg: dict, run_ts: str | None = None) -> dict:
    # genera rutas con timestamp y las añade a cfg["logging_ext"]
    ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = Path(cfg["logging"]["runs_dir"])
    reps_root = Path(cfg["logging"]["reports_dir"])

    run_dir = runs_root / f"run_{ts}"
    rep_dir = reps_root / f"run_{ts}"
    ensureDir(run_dir)
    ensureDir(rep_dir)

    cfg_ext = dict(cfg)  # copia superficial
    cfg_ext["logging_ext"] = {
        "run_ts": ts,
        "run_dir": str(run_dir),
        "report_dir": str(rep_dir),
    }
    return cfg_ext


def loadYamlConfig(path_yaml: str, attach_run_dirs: bool = True) -> dict:
    path = Path(path_yaml)
    if not path.exists():
        raise FileNotFoundError(f"no existe el yaml: {path_yaml}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("el yaml debe mapear a un dict de nivel raíz")

    checkNoExtraKeys(data, REQUIRED_SCHEMA, root="config")
    checkMissingKeys(data, REQUIRED_SCHEMA, root="config")
    cfg = validateAndCoerce(data, REQUIRED_SCHEMA, root="config")

    if attach_run_dirs:
        cfg = deriveRunDirs(cfg)

    # guardar config efectiva dentro del directorio de reportes del run
    rep_dir_str = cfg.get("logging_ext", {}).get("report_dir", cfg["logging"]["reports_dir"])
    ensureDir(Path(rep_dir_str))
    out_cfg = Path(rep_dir_str) / cfg["logging"]["run_config_used"]
    writeYaml(out_cfg, cfg)
    return cfg
