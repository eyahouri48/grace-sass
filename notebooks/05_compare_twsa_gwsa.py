# fichier : notebooks/05_compare_twsa_gwsa.py
"""Compare TWSA brute (×10, mm) et proxy GWSA pour montrer
que la soustraction GLDAS change peu sur le SASS."""

import pandas as pd
import plotly.graph_objects as go
from pipeline.config import DATA_DIR

# Charger le cache principal
df = pd.read_parquet(DATA_DIR / "sass_series.parquet")

fig = go.Figure()

# TWSA brute convertie en mm (pour comparaison directe)
fig.add_trace(go.Scatter(
    x=df.index, y=df["twsa_cm"] * 10,
    name="TWSA brute (×10, mm)",
    line=dict(color="steelblue", width=2),
))

# Proxy GWSA
fig.add_trace(go.Scatter(
    x=df.index, y=df["gwsa_mm"],
    name="Proxy GWSA (mm)",
    line=dict(color="darkorange", width=2, dash="dash"),
))

# Composante GLDAS soustraite (pour montrer qu'elle est faible)
fig.add_trace(go.Scatter(
    x=df.index, y=df["gldas_anom_mm"],
    name="Anomalie GLDAS soustraite (mm)",
    line=dict(color="gray", width=1),
    opacity=0.6,
))

fig.update_layout(
    title="TWSA brute vs. proxy GWSA<br>",
    xaxis_title="",
    yaxis_title="Anomalie de stockage (mm)",
    legend=dict(x=0, y=-0.2, orientation="h"),
    template="plotly_white",
    height=500,
)


fig.write_html("notebooks/compare_twsa_gwsa.html")
fig.write_image("notebooks/compare_twsa_gwsa.png", width=1000, height=500)
print("Graphiques sauvegardés : .html + .png")