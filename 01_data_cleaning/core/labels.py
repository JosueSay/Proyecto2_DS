import pandas as pd

def generateLabel(df: pd.DataFrame) -> pd.DataFrame:
    # mapea cada fila a una etiqueta según el ganador
    def mapLabel(row):
        tie = int(row.get("winner_tie", 0))
        a = int(row.get("winner_model_a", 0))
        b = int(row.get("winner_model_b", 0))
        
        # si hay más de un ganador o ninguno, marcar como NA
        if tie + a + b != 1:
            return pd.NA
        
        # empate -> etiqueta 2
        if tie == 1:
            return 2
        
        # si no hay empate, devolver 0 para a, 1 para b
        return 0 if a == 1 else 1

    # crear columna label aplicando el mapeo
    df["label"] = df.apply(mapLabel, axis=1)
    
    # filtrar filas con label válido
    return df[df["label"].notna()].copy()
