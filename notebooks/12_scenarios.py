# fichier : notebooks/12_scenarios.py
"""Exploration Tâche 12 — Cadrage par scénarios."""

import pandas as pd
import plotly.graph_objects as go
from pipeline.scenarios import build_scenarios, get_scenario_summary

# --- Charger la série depuis le cache ---
cache = pd.read_parquet("data/sass_series.parquet")
gwsa = cache["gwsa_mm"]
gwsa.index = pd.to_datetime(gwsa.index)
gwsa.index.name = "date"
gwsa.name = "gwsa_mm"

# --- MAE validée (résultat de la Tâche 10) ---
VALIDATED_MAE = 6.1  # mm — issue de la CV Prophet

# --- Construire les scénarios ---
scenarios = build_scenarios(gwsa, validated_mae_mm=VALIDATED_MAE)

# --- Afficher le tableau de jalons ---
summary = get_scenario_summary(scenarios)
print("\n Tableau de synthèse des scénarios :")
print(summary.to_string(index=False))

# --- Afficher les avertissements ---
print("\n Avertissements :")
for lang in ("fr", "en"):
    print(f"\n  [{lang.upper()}]")
    for zone, text in scenarios["warnings"][lang].items():
        print(f"    {zone}: {text}")

# --- Vérifier la structure ---
fdf = scenarios["forecast_df"]
print(f"\n Forme du DataFrame prévision : {fdf.shape}")
print(f"   Zones : {fdf['zone'].value_counts().to_dict()}")
print(f"   Dernière obs : {scenarios['last_obs_date']}")
print(f"   Coupure validé/extrapolation : {scenarios['cutoff_date']}")

# --- Graphique : fan chart avec zones ---

# Découper le DataFrame prévision en deux zones
validated = fdf[fdf["zone"] == "validated"]
extrap = fdf[fdf["zone"] == "extrapolation"]

# Conversion globale : toutes les dates en datetime Python natif pour Kaleido
hist_dates = gwsa.index.to_pydatetime()
val_dates = validated["ds"].dt.to_pydatetime()
ext_dates = extrap["ds"].dt.to_pydatetime()

fig = go.Figure()

# Série historique
fig.add_trace(go.Scatter(
    x=hist_dates, y=gwsa.values,
    mode="lines", name="Observé (gwsa_mm)",
    line=dict(color="#1f4e79", width=2),
))

# Zone validée — ligne
fig.add_trace(go.Scatter(
    x=val_dates, y=validated["yhat"].values,
    mode="lines", name="Prévision validée (0–24 mois)",
    line=dict(color="#2e75b6", width=2),
))
# Zone validée — bande IC
fig.add_trace(go.Scatter(
    x=list(val_dates) + list(val_dates)[::-1],
    y=list(validated["yhat_upper"]) + list(validated["yhat_lower"])[::-1],
    fill="toself", fillcolor="rgba(46,117,182,0.2)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))

# Zone extrapolation — ligne
fig.add_trace(go.Scatter(
    x=ext_dates, y=extrap["yhat"].values,
    mode="lines", name="Extrapolation scénario (24–60 mois)",
    line=dict(color="#c55a11", width=2, dash="dash"),
))
# Zone extrapolation — bande IC
fig.add_trace(go.Scatter(
    x=list(ext_dates) + list(ext_dates)[::-1],
    y=list(extrap["yhat_upper"]) + list(extrap["yhat_lower"])[::-1],
    fill="toself", fillcolor="rgba(197,90,17,0.15)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))

# Ligne de coupure
fig.add_vline(
    x=scenarios["cutoff_date"].to_pydatetime(), line_dash="dot", line_color="gray",
    annotation_text="Fin horizon validé", annotation_position="top left",
)

fig.update_layout(
    title="SASS — Prévision validée vs. extrapolation de scénario",
    xaxis_title="Date",
    yaxis_title="Anomalie de stock d'eau souterraine (mm)",
    template="plotly_white",
    legend=dict(
        yanchor="bottom", y=0.01,
        xanchor="left", x=0.01,
        bgcolor="rgba(255,255,255,0.8)",
    ),
)

fig.show()  # Ouvre le graphique dans le navigateur automatiquement
fig.write_html("notebooks/scenario_fan_chart.html")
fig.write_image("notebooks/scenario_fan_chart.png", width=1100, height=500)
print("\n Fan chart sauvegardé : notebooks/scenario_fan_chart.png")