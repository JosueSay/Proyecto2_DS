# límites reales de tokenización
MAX_LEN_PROMPT = 96
MAX_LEN_RESP = 288
TOTAL_BUDGET = 387

# acolchonamiento y caps
SPECIAL_TOK_PAD = 9
RESP_MIN = 120
RESP_MAX = 160
BULLET_CAP = 24

# extracción/truncado mixto
HEAD_TOK = 96
TAIL_TOK = 32
MID_STRIDE = 24

def computeRespTarget(prompt_len: int) -> int:
    # target dinámico por respuesta con clip al rango seguro
    usable = TOTAL_BUDGET - min(int(prompt_len), MAX_LEN_PROMPT) - SPECIAL_TOK_PAD
    target = usable / 2.0
    target = max(RESP_MIN, min(RESP_MAX, target))
    return int(round(target))
