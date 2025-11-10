from pathlib import Path
from typing import Any
import json
import yaml
import pandas as pd


def ensureDir(path: str | Path) -> None:
    # crea el directorio si no existe
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def readCsv(path: str | Path) -> pd.DataFrame:
    # lee csv con fallback de encoding
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no existe el archivo: {p}")
    try:
        return pd.read_csv(p, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(p, engine="python", encoding="latin-1")


def writeCsv(path: str | Path, df: pd.DataFrame, index: bool = False) -> None:
    # escribe csv creando directorios
    p = Path(path)
    ensureDir(p.parent)
    df.to_csv(p, index=index)


def readYaml(path: str | Path) -> dict:
    # lee yaml y devuelve dict
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no existe el archivo: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def writeYaml(path: str | Path, data: Any) -> None:
    # escribe yaml creando directorios
    p = Path(path)
    ensureDir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def readJson(path: str | Path) -> Any:
    # lee json y devuelve objeto python
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no existe el archivo: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def writeJson(path: str | Path, data: Any, indent: int = 2) -> None:
    # escribe json creando directorios
    p = Path(path)
    ensureDir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
