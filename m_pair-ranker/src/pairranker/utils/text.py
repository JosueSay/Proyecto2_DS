import re
import ast
import math
import pandas as pd


# patrones para limpiar unicode problemático
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def sanitizeText(s: str) -> str:
    # normaliza espacios y remueve caracteres de control
    s = "" if s is None else str(s)
    s = SURROGATE_RE.sub(" ", s)
    s = CONTROL_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def toStrSafe(x) -> str:
    # convierte a str evitando NaN, listas serializadas y bytes
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
        if pd.isna(x):
            return ""
    except Exception:
        pass

    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", "ignore")
        except Exception:
            x = x.decode("latin-1", "ignore")

    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            arr = ast.literal_eval(x)
            if isinstance(arr, list):
                x = "\n".join(map(str, arr))
        except Exception:
            pass

    return sanitizeText(x)
