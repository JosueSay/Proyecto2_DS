import matplotlib.pyplot as plt

PALETTE = {
    "dominante": "#1B3B5F",
    "secundario": "#2E5984",
    "mediacion": "#5C7EA3",
    "neutro": "#B0BEC5",
    "acento": "#F28C38",
    "confirmacion": "#3D8361",
    "advertencia": "#C14953",
    "fondo": "#F7F9FC",
}

def setMplStyle():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlecolor": PALETTE["dominante"],
        "axes.labelcolor": PALETTE["dominante"],
        "xtick.color": PALETTE["dominante"],
        "ytick.color": PALETTE["dominante"],
        "grid.color": PALETTE["neutro"],
        "grid.alpha": 0.12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["dominante"],
    })
