import re
import pandas as pd

# intentar cargar stopwords de nltk
try:
    import nltk
    nltk.data.find("corpora/stopwords")
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass

# crear set de stopwords si se puede
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()

def evalAndJoin(text: str):
    if pd.isna(text):
        return None
    t = str(text).strip()
    if t.lower() in {"", "nan", "null", "none"}:
        return None
    t = t.replace(r"\/", "/").replace(r"null", '""')
    try:
        arr = eval(t)  # evaluar string como lista o valor python
        if isinstance(arr, list):
            # limpiar elementos vacíos y unir con saltos de línea
            arr = [str(a).strip() for a in arr if str(a).strip()]
            return "\n".join(arr) if arr else None
        if arr is None:
            return None
        return str(arr)
    except Exception:
        return t or None

def cleanText(text: str, remove_stopwords: bool, keep_punct: bool) -> str:
    if not isinstance(text, str):
        return None
    txt = text.lower()
    txt = re.sub(r"(https?://\S+|www\.\S+)", " ", txt)  # quitar urls
    txt = re.sub(r"@\w+", " ", txt)  # quitar menciones
    txt = re.sub(r"#\w+", " ", txt)  # quitar hashtags
    if keep_punct:
        txt = re.sub(r"[^\w\s\.\,\!\?\;\:]", " ", txt)  # mantener puntuación básica
    else:
        txt = re.sub(r"[^\w\s]", " ", txt)  # quitar toda la puntuación
    txt = re.sub(r"\s+", " ", txt).strip()  # limpiar espacios extra
    if remove_stopwords and STOPWORDS:
        # quitar stopwords si se pidió
        txt = " ".join(w for w in txt.split() if w not in STOPWORDS).strip()
    return txt or None

def removeInvalidChars(text: str) -> str:
    # reemplazar chars inválidos por unicode seguro
    return text.encode("utf-8", errors="replace").decode("utf-8")

def normalizeDigits(s: str) -> str:
    import re as _re
    return _re.sub(r"\d+", "<num>", s)  # reemplazar números por <num>

def promptSig(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\t\r\n]+", " ", s)  # quitar saltos de línea/tab
    s = re.sub(r"[^\w\s\.\,\!\?\;\:]", " ", s)  # limpiar chars raros
    s = re.sub(r"\s+", " ", s).strip()  # limpiar espacios extra
    s = normalizeDigits(s)  # normalizar números
    return s

def infoLen(text: str) -> int:
    t = str(text or "").strip()
    if t == "no response":
        return 0
    return len(t.split())  # contar palabras
