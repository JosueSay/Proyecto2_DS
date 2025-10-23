import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
SUMMARY_LOG = os.path.join(LOG_DIR, "data_summary.log")

def showSummary():
    """
    Retorna el contenido del log de resumen de datos como string.
    Si no existe el log, retorna None.
    """
    if os.path.exists(SUMMARY_LOG):
        with open(SUMMARY_LOG, "r", encoding="utf-8") as f:
            return f.read()
    return None

if __name__ == "__main__":
    content = showSummary()
    if content is not None:
        print(content)
