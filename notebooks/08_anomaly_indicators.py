# fichier : notebooks/08_anomaly_indicators.py
"""Exploration des indicateurs d'anomalie sur la série SASS réelle."""

import pandas as pd
from pipeline.indicators import compute_anomaly_indicators
from pipeline.config import SERIES_PARQUET

df = pd.read_parquet(SERIES_PARQUET)

# --- Calculer z-score et percentile ---
result = compute_anomaly_indicators(df)

# --- Afficher les derniers mois ---
print("=== 10 derniers mois ===")
print(result[["gwsa_mm", "is_imputed", "zscore", "percentile_rank"]].tail(10))

# --- Stats rapides sur la référence 2004-2009 ---
baseline = result.loc["2004-01":"2009-12"]
baseline_obs = baseline[~baseline["is_imputed"]]
print(f"\nMoyenne z-score baseline (observés) : {baseline_obs['zscore'].mean():.6f}")
print(f"Nb mois baseline observés : {len(baseline_obs)}")

# --- Dernier mois disponible ---
last = result.dropna(subset=["gwsa_mm"]).iloc[-1]
print(f"\nDernier mois : {last.name}")
print(f"  gwsa_mm        = {last['gwsa_mm']:.1f} mm")
print(f"  z-score        = {last['zscore']:.2f}")
print(f"  percentile     = {last['percentile_rank']:.1f} %")

# heatmap calendaire ---
import plotly.express as px

zs = result["zscore"].copy()
zs_pivot = zs.groupby([zs.index.year, zs.index.month]).first().unstack()
zs_pivot.columns = ["Jan","Fév","Mar","Avr","Mai","Jun",
                     "Jul","Aoû","Sep","Oct","Nov","Déc"]

fig = px.imshow(
    zs_pivot,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    aspect="auto",
    labels={"color": "Z-score"},
    title="Anomalie de stock d'eau souterraine du SASS — z-score mensuel (réf. 2004–2009)",
)
fig.update_yaxes(title="Année", dtick=1)
fig.update_xaxes(title="Mois", side="top")
fig.show()