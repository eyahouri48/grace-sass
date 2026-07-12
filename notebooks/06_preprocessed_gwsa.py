# fichier : notebooks/06_preprocessed_gwsa.py

import logging
import plotly.graph_objects as go
from pipeline.preprocessing import load_proxy_series, reindex_monthly, interpolate_gaps, find_gap_periods, add_uncertainty_band
from pipeline.config import COLORS, AQUIFER_ID

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Charger la série prétraitée
df = load_proxy_series()
df = reindex_monthly(df)
df = interpolate_gaps(df)
gaps = find_gap_periods(df)
df = add_uncertainty_band(df)

# Séparer observés et imputés
obs = df[~df["is_imputed"]]
imp = df[df["is_imputed"]]

fig = go.Figure()

# Bande d'incertitude (±uncertainty_mm autour de gwsa_mm)
fig.add_trace(go.Scatter(
    x=list(df.index) + list(df.index[::-1]),
    y=list(df["gwsa_mm"] + df["uncertainty_mm"])
      + list((df["gwsa_mm"] - df["uncertainty_mm"])[::-1]),
    fill="toself",
    fillcolor=COLORS["band"],
    line=dict(width=0),
    name="Incertitude mascon (indicative)",
    showlegend=True,
    hoverinfo="skip",
))

# Mois observés — trait plein
fig.add_trace(go.Scatter(
    x=obs.index, y=obs["gwsa_mm"],
    mode="lines",
    line=dict(color=COLORS["primary"], width=2),
    name="GWSA observé",
))

# Mois imputés — points gris
fig.add_trace(go.Scatter(
    x=imp.index, y=imp["gwsa_mm"],
    mode="markers",
    marker=dict(color=COLORS["context"], size=4),
    name="Mois imputés (interpolés)",
))

# Zone grisée pour la lacune inter-missions
for start, end, n in gaps:
    if n >= 8:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=COLORS["context"], opacity=0.15,
            line_width=0,
            annotation_text="Lacune inter-missions",
            annotation_position="top",
            annotation_font_size=10,
        )

fig.update_layout(
    title=f"Proxy GWSA prétraité — {AQUIFER_ID.upper()} (avr 2002 → mar 2026)",
    xaxis_title="Date",
    yaxis_title="GWSA (mm, anomalie vs 2004–2009)",
    template="plotly_white",
    legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)"),
    height=500,
) 

fig.write_html("notebooks/preprocessed_gwsa.html", include_plotlyjs="cdn")
fig.show()
print("Graphique sauvegardé : notebooks/preprocessed_gwsa.html")