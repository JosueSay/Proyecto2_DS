import matplotlib.pyplot as plt

PALETTE = {
    "dominante":   "#1B3B5F",
    "secundario":  "#2E5984",
    "mediacion":   "#5C7EA3",
    "neutro":      "#B0BEC5",
    "acento":      "#F28C38",
    "confirmacion":"#3D8361",
    "advertencia": "#C14953",
    "fondo":       "#F7F9FC",
}

# kwargs de exportación consistentes
EXPORT_KW = dict(dpi=300, bbox_inches="tight", facecolor="white")

def setMplStyle():
    # estilo base consistente con 02_eda
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "axes.titlecolor": PALETTE["dominante"],
        "axes.labelcolor": PALETTE["dominante"],
        "xtick.color": PALETTE["dominante"],
        "ytick.color": PALETTE["dominante"],
        "text.color":  PALETTE["dominante"],

        "axes.grid": True,
        "grid.color": PALETTE["neutro"],
        "grid.linestyle": "-",
        "grid.alpha": 0.12,

        "lines.linewidth": 2.0,
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.frameon": False,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["dominante"],
        "axes.linewidth": 1.0,
    })

def getModelColor(name: str) -> str:
    # asigna color estable por nombre de modelo
    key = (name or "").lower()
    if "deberta" in key: return PALETTE["confirmacion"]
    if "roberta" in key: return PALETTE["secundario"]
    if "electra" in key: return PALETTE["mediacion"]
    if "xlnet" in key:   return PALETTE["acento"]
    return PALETTE["dominante"]

def saveFig(path: str):
    # wrapper de guardado con EXPORT_KW
    plt.tight_layout()
    plt.savefig(path, **EXPORT_KW)
    plt.close()
