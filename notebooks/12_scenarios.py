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

fig = go.Figure()

# Série historique
fig.add_trace(go.Scatter(
    x=gwsa.index, y=gwsa.values,
    mode="lines", name="Observé (gwsa_mm)",
    line=dict(color="#1f4e79", width=2),
))

# Zone validée
validated = fdf[fdf["zone"] == "validated"]
fig.add_trace(go.Scatter(
    x=validated["ds"], y=validated["yhat"],
    mode="lines", name=f"Prévision validée (0–24 mois)",
    line=dict(color="#2e75b6", width=2),
))
fig.add_trace(go.Scatter(
    x=pd.concat([validated["ds"], validated["ds"][::-1]]),
    y=pd.concat([validated["yhat_upper"], validated["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(46,117,182,0.2)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))

# Zone extrapolation
extrap = fdf[fdf["zone"] == "extrapolation"]
fig.add_trace(go.Scatter(
    x=extrap["ds"], y=extrap["yhat"],
    mode="lines", name=f"Extrapolation scénario (24–60 mois)",
    line=dict(color="#c55a11", width=2, dash="dash"),
))
fig.add_trace(go.Scatter(
    x=pd.concat([extrap["ds"], extrap["ds"][::-1]]),
    y=pd.concat([extrap["yhat_upper"], extrap["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(197,90,17,0.15)",
    line=dict(width=0), showlegend=False, hoverinfo="skip",
))

# Ligne de coupure
fig.add_vline(
    x=scenarios["cutoff_date"], line_dash="dot", line_color="gray",
    annotation_text="Fin horizon validé", annotation_position="top left",
)

fig.update_layout(
    title="SASS — Prévision validée vs. extrapolation de scénario",
    xaxis_title="Date",
    yaxis_title="Anomalie de stock d'eau souterraine (mm)",
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)

fig.write_html("notebooks/scenario_fan_chart.html")
fig.write_image("notebooks/scenario_fan_chart.png", width=1100, height=500)
print("\n Fan chart sauvegardé : notebooks/scenario_fan_chart.png")