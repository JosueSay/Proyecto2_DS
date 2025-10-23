import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "00_cache-manager"))
from cache_manager import CacheManager

# Descargar stopwords
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# === Configuración rutas ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CACHE_DIR = os.path.join(BASE_DIR, "..", "cache")
CACHE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "00_cache-manager", "cache_config.yaml")

cacheManager = CacheManager(CACHE_CONFIG, CACHE_DIR)
CACHE_KEY = "CleaningDone"

# === Funciones de limpieza y preparación ===
def generateLabel(df: pd.DataFrame) -> pd.DataFrame:
    def mapLabel(row):
        tie = int(row.get('winner_tie', 0))
        a = int(row.get('winner_model_a', 0))
        b = int(row.get('winner_model_b', 0))
        if tie + a + b != 1:
            return pd.NA
        if tie == 1:
            return 2
        elif a == 1:
            return 0
        else:
            return 1
    df['label'] = df.apply(mapLabel, axis=1)
    return df[df['label'].notna()]

def evalAndJoin(text):
    if pd.isna(text):
        return pd.NA
    if isinstance(text, str):
        t = text.strip().lower()
        if t in ["nan", "null", "none", ""]:
            return pd.NA
    try:
        arr = eval(text)
        if isinstance(arr, list):
            arr = [str(a).strip() for a in arr if str(a).strip().lower() not in ["nan", "null", "none"]]
            return "\n".join(arr) if arr else pd.NA
        elif arr is None:
            return pd.NA
        else:
            return str(arr)
    except Exception:
        return str(text).strip() or pd.NA

def cleanText(text: str) -> str:
    if not isinstance(text, str):
        return None
    text = text.lower()
    text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[_\*\~\^\(\)\[\]\{\}\|]", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(w for w in text.split() if w not in STOPWORDS).strip()
    return text if text else None

def removeInvalidChars(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")

def cleanDataFrame(df: pd.DataFrame) -> pd.DataFrame:
    df["prompt_clean"] = df["prompt"].apply(evalAndJoin).apply(cleanText)
    df["response_a_clean"] = df["response_a"].apply(evalAndJoin).apply(cleanText)
    df["response_b_clean"] = df["response_b"].apply(evalAndJoin).apply(cleanText)

    df = df.dropna(subset=["prompt_clean", "response_a_clean", "response_b_clean"]).copy()

    for col in ["prompt_clean", "response_a_clean", "response_b_clean"]:
        df[col] = df[col].apply(removeInvalidChars)

    df = df[
        (df["prompt_clean"].str.strip() != "") &
        (df["response_a_clean"].str.strip() != "") &
        (df["response_b_clean"].str.strip() != "")
    ].copy()

    return df

def createReverseDf(df: pd.DataFrame) -> pd.DataFrame:
    dfReverse = df.copy()
    dfReverse["response_a_clean"], dfReverse["response_b_clean"] = df["response_b_clean"], df["response_a_clean"]
    dfReverse["label"] = dfReverse["label"].map({0:1, 1:0, 2:2})
    dfReverse["reverse"] = True
    df["reverse"] = False
    dfFull = pd.concat([df, dfReverse], axis=0)

    dfFull = dfFull.dropna(subset=["prompt_clean", "response_a_clean", "response_b_clean"])
    dfFull = dfFull[
        (dfFull["prompt_clean"].str.strip() != "") &
        (dfFull["response_a_clean"].str.strip() != "") &
        (dfFull["response_b_clean"].str.strip() != "")
    ].copy()

    for col in ["prompt_clean", "response_a_clean", "response_b_clean"]:
        dfFull[col] = dfFull[col].apply(removeInvalidChars)

    return dfFull

# === Main ===
def main():
    if cacheManager.exists(CACHE_KEY):
        print("CACHE_USED")
        sys.exit(0)

    inputPath = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(inputPath):
        raise FileNotFoundError(f"No se encontró el archivo\t{inputPath}")

    print("Cargando dataset original...")
    df = pd.read_csv(inputPath)
    print(f"\nRegistros originales:\t{len(df)}")

    print("\tLimpiando datos...")
    dfClean = cleanDataFrame(df)
    print(f"Registros después de limpieza:\t{len(dfClean)}")

    print("\tGenerando labels...")
    dfClean = generateLabel(dfClean)

    print("\tGenerando reverses (swap A-B)...")
    dfClean = createReverseDf(dfClean)
    print(f"Registros después de reverse:\t{len(dfClean)}")


    # Limpiar caracteres inválidos
    for col in ["prompt_clean", "response_a_clean", "response_b_clean"]:
        dfClean[col] = dfClean[col].apply(removeInvalidChars)

    # Dividir dataset (estratificado por label)
    print("\tDividiendo dataset en train / validation...")
    trainDf, valDf = train_test_split(dfClean, test_size=0.3, random_state=42, stratify=dfClean["label"])

    # Guardar CSVs con manejo de Unicode
    trainDf.to_csv(os.path.join(DATA_DIR, "train_clean.csv"), index=False, encoding="utf-8-sig", errors="replace")
    valDf.to_csv(os.path.join(DATA_DIR, "validation_clean.csv"), index=False, encoding="utf-8-sig", errors="replace")

    print(f"\nArchivos generados:")
    print(f"\t- train_clean.csv\t({len(trainDf)} filas)")
    print(f"\t- validation_clean.csv\t({len(valDf)} filas)")

    # Crear cache
    cacheManager.create(CACHE_KEY, content="Data cleaned successfully.")
    print("DONE")
    sys.exit(0)

if __name__ == "__main__":
    main()
