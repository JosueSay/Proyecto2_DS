import re
import pandas as pd

def estimateTruncation(len_series: pd.Series, max_len: int) -> pd.Series:
    # devuelve True donde la serie excede max_len
    return (len_series > max_len)

def collapseBulletsAndRepeats(text: str, max_bullets: int = 40) -> str:
    t = str(text or "")  # asegurar que sea string
    lines = [l for l in t.split("\n")]
    count = 0
    out_lines = []
    for l in lines:
        # contar bullets y limitar a max_bullets
        if re.match(r"^\s*(?:[-\*\•]|\d+\.)\s+", l):
            count += 1
            if count <= max_bullets:
                out_lines.append(l)
        else:
            out_lines.append(l)
    if count > max_bullets:
        out_lines.append("…")  # indicar truncamiento
    t = "\n".join(out_lines)

    # separar en chunks por saltos de línea dobles
    chunks = [c.strip() for c in re.split(r"(\n\s*\n+)", t)]
    seen = {}
    new_chunks = []
    for c in chunks:
        if not c.strip():
            new_chunks.append(c)
            continue
        k = c.lower()
        seen[k] = seen.get(k, 0) + 1
        if seen[k] <= 2:  # permitir máximo 2 repeticiones
            new_chunks.append(c)
    t = "".join(new_chunks)

    # quitar saludos largos
    t = re.sub(r"^(?:hi|hello|dear|greetings)[\s\.,!\-:]*.{150,300}?(?=\n|$)", "", t, flags=re.I|re.S)
    # quitar menciones de ser IA
    t = re.sub(r"(?:as an ai|i am an ai|language model).*", "", t, flags=re.I)
    return t.strip() or text  # si queda vacío, devolver texto original

def applyTailControl(df: pd.DataFrame, max_len_resp=512, p99_hint: dict = None):
    # extraer percentiles p99 si hay hint
    p99_a = p99_hint.get("respA_len_p99") if p99_hint else None
    p99_b = p99_hint.get("respB_len_p99") if p99_hint else None

    # calcular largo de cada respuesta
    a_len = df["response_a_clean"].str.split().str.len()
    b_len = df["response_b_clean"].str.split().str.len()

    # determinar qué filas necesitan truncamiento
    need_a = estimateTruncation(a_len, max_len_resp) | (a_len >= (p99_a or a_len.quantile(0.99)))
    need_b = estimateTruncation(b_len, max_len_resp) | (b_len >= (p99_b or b_len.quantile(0.99)))

    # truncar donde se necesita
    df.loc[need_a, "response_a_clean"] = df.loc[need_a, "response_a_clean"].map(lambda t: collapseBulletsAndRepeats(t, 40))
    df.loc[need_b, "response_b_clean"] = df.loc[need_b, "response_b_clean"].map(lambda t: collapseBulletsAndRepeats(t, 40))
    
    return df
