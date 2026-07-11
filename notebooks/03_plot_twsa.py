# fichier : notebooks/03_plot_twsa.py
# Premier graphique de la série TWSA brute


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from pipeline.config import DATA_DIR

# Charger le cache
twsa = pd.read_parquet(DATA_DIR / "twsa_cm.parquet")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(twsa.index, twsa["twsa_cm"], color="#1a4e8a", linewidth=0.8)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="Référence 2004–2009")
ax.set_title("TWSA brute sur le SASS — mascon JPL CRI (cm)")
ax.set_ylabel("Anomalie de stockage (cm, LWE)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("notebooks/twsa_brute.png", dpi=150)
print("→ Graphique sauvegardé : notebooks/twsa_brute.png")
plt.show()