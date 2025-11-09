import re
import pandas as pd

from settings import (
    MAX_LEN_PROMPT, MAX_LEN_RESP, BULLET_CAP,
    HEAD_TOK, TAIL_TOK, MID_STRIDE, computeRespTarget
)
from text_utils import splitSentences, tokenCount, stripBoilerplate

def estimateTruncation(len_series: pd.Series, max_len: int) -> pd.Series:
    return (len_series > max_len)

def collapseBulletsAndRepeats(text: str, max_bullets: int = BULLET_CAP) -> str:
    t = str(text or "")
    # limitar bullets
    lines = t.split("\n")
    out, seen, count = [], {}, 0
    for l in lines:
        if re.match(r"^\s*(?:[-\*\•]|\d+\.)\s+", l):
            count += 1
            if count <= max_bullets:
                out.append(l)
        else:
            out.append(l)
    if count > max_bullets:
        out.append("…")
    t = "\n".join(out)

    # dedup de bloques repetidos (>2)
    chunks = [c for c in re.split(r"(\n\s*\n+)", t)]
    out = []
    for c in chunks:
        k = c.strip().lower()
        if k and not k.isspace():
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= 2:
                out.append(c)
        else:
            out.append(c)
    return "".join(out).strip() or text

def headTailTokens(tokens: list, head: int, tail: int) -> list:
    if len(tokens) <= head + tail:
        return tokens
    return tokens[:head] + ["…"] + tokens[-tail:]

def sampleMiddle(tokens: list, target: int, stride: int = MID_STRIDE) -> list:
    if len(tokens) <= target:
        return tokens
    # recorta del centro con muestreo
    left = tokens[: len(tokens)//2]
    right = tokens[len(tokens)//2 :]
    mid = tokens[len(left)-min(64, len(tokens)//4) : len(left)+min(64, len(tokens)//4)]
    sampled = mid[::max(1, stride)]
    merged = left[:HEAD_TOK] + ["…"] + sampled + ["…"] + right[-TAIL_TOK:]
    if len(merged) > target:
        merged = merged[:target-1] + ["…"]
    return merged

def compressToTarget(text: str, resp_target: int) -> str:
    t = stripBoilerplate(str(text or "").strip())
    if not t:
        return text

    # si cabe, retorna
    toks = t.split()
    if len(toks) <= resp_target:
        return t

    # head+tail
    toks = headTailTokens(toks, HEAD_TOK, TAIL_TOK)
    if len(toks) <= resp_target:
        return " ".join(toks)

    # muestreo del medio
    toks = sampleMiddle(toks, resp_target, MID_STRIDE)
    # dedup de oraciones exactas (>2)
    sents = splitSentences(" ".join(toks))
    counts, out = {}, []
    for s in sents:
        k = s.strip().lower()
        if not k:
            continue
        counts[k] = counts.get(k, 0) + 1
        if counts[k] <= 2:
            out.append(s)
    out_text = " ".join(out).strip()
    if tokenCount(out_text) > resp_target:
        out_text = " ".join(out_text.split()[:resp_target-1] + ["…"])
    return out_text or text

def miniHeadTailPrompt(s: str) -> str:
    toks = str(s or "").split()
    if len(toks) <= MAX_LEN_PROMPT:
        return s
    toks = headTailTokens(toks, HEAD_TOK // 2, TAIL_TOK // 2)
    return " ".join(toks)

def applyTailControl(df: pd.DataFrame, max_len_resp: int = MAX_LEN_RESP, p99_hint: dict | None = None):
    # longitudes base
    a_len = df["response_a_clean"].astype(str).str.split().str.len()
    b_len = df["response_b_clean"].astype(str).str.split().str.len()
    p_len = df["prompt_clean"].astype(str).str.split().str.len()

    # p99 opcional
    p99_a = p99_hint.get("respA_len_p99") if p99_hint else a_len.quantile(0.99)
    p99_b = p99_hint.get("respB_len_p99") if p99_hint else b_len.quantile(0.99)

    need_a = estimateTruncation(a_len, max_len_resp) | (a_len >= p99_a)
    need_b = estimateTruncation(b_len, max_len_resp) | (b_len >= p99_b)

    # objetivo dinámico por fila
    resp_targets = p_len.map(lambda n: computeRespTarget(int(n)))

    # A
    if need_a.any():
        sub = df.loc[need_a, "response_a_clean"].map(lambda t: collapseBulletsAndRepeats(t, BULLET_CAP))
        sub = sub.map(stripBoilerplate)
        df.loc[need_a, "response_a_clean"] = [
            compressToTarget(t, int(resp_targets.loc[idx])) for idx, t in sub.items()
        ]

    # B
    if need_b.any():
        sub = df.loc[need_b, "response_b_clean"].map(lambda t: collapseBulletsAndRepeats(t, BULLET_CAP))
        sub = sub.map(stripBoilerplate)
        df.loc[need_b, "response_b_clean"] = [
            compressToTarget(t, int(resp_targets.loc[idx])) for idx, t in sub.items()
        ]

    # prompt mini head+tail si excede
    over_p = p_len > MAX_LEN_PROMPT
    if over_p.any():
        df.loc[over_p, "prompt_clean"] = df.loc[over_p, "prompt_clean"].map(miniHeadTailPrompt)

    return df
