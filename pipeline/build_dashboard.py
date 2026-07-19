# fichier : pipeline/build_dashboard.py

"""
Génère le dashboard statique HTML (Path A — GitHub Pages).

Lit le cache Parquet, calcule les KPI, crée les figures Plotly,
et assemble le tout dans docs/index.html.
"""

import json


import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from pipeline import config
from pipeline.scenarios import build_scenarios

# ── Fonctions utilitaires ──────────────────────────────────────

def load_strings(lang: str = "en") -> dict:
    """Charge les libellés d'interface pour la langue donnée."""
    path = config.UI_STRINGS_DIR / f"{lang}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data() -> pd.DataFrame:
    """Charge le cache Parquet principal."""
    return pd.read_parquet(config.SERIES_PARQUET)


# ── Figure : série temporelle gwsa_mm ──────────────────────────

def make_timeseries_figure(df: pd.DataFrame, strings: dict) -> go.Figure:
    """
    Graphique principal du dashboard : série gwsa_mm historique.

    Trois couches visuelles :
    - fond : rectangle grisé sur la lacune 2017-2018
    - couche basse : série complète (interpolée) en pointillés gris
    - couche haute : mois observés seulement en trait plein bleu

    Paramètres
    ----------
    df : DataFrame avec colonnes 'gwsa_mm' et 'is_imputed', index DatetimeIndex
    strings : dictionnaire des libellés (depuis ui_strings/*.json)

    Retourne
    --------
    go.Figure prête à être convertie en HTML
    """
    fig = go.Figure()

    # ── Couche fond : zone grisée pour la lacune 2017-2018 ──
    # add_vrect dessine un rectangle vertical semi-transparent
    # derrière toutes les traces (layer="below")
    fig.add_vrect(
        x0="2017-06-01",
        x1="2018-06-01",
        fillcolor=config.COLORS["gap_zone"],
        opacity=0.8,
        layer="below",
        line_width=0,
        # Annotation discrète en haut du rectangle
        annotation_text=strings["ts_gap_label"],
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color=config.COLORS["neutral_mid"],
    )

    # ── Couche basse : série complète en pointillés gris ──
    # On trace TOUTE la série (y compris les mois interpolés).
    # Les segments interpolés apparaîtront en gris pointillé.
    # Les segments observés seront masqués par la trace bleue
    # qu'on ajoute par-dessus juste après.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["gwsa_mm"],
        mode="lines",
        name=strings["ts_imputed"],
        line=dict(
            color=config.COLORS["imputed"],
            width=1.2,
            dash="dot",          # pointillés
        ),
        hoverinfo="skip",        # pas de tooltip sur cette couche
        showlegend=True,
    ))

    # ── Couche haute : mois observés en trait plein bleu ──
    # On ne garde que les mois NON imputés.
    # Ils se dessinent par-dessus les pointillés.
    obs = df[~df["is_imputed"]].copy()

    fig.add_trace(go.Scatter(
        x=obs.index,
        y=obs["gwsa_mm"],
        mode="lines",
        name=strings["ts_observed"],
        line=dict(
            color=config.COLORS["primary"],
            width=1.8,
        ),
        # Tooltip clair : "Mar 2015 : -42.3 mm"
        hovertemplate="%{x|%b %Y} : %{y:.1f} mm<extra></extra>",
    ))

    # ── Mise en forme générale ──
    fig.update_layout(
        yaxis_title=strings["ts_ylabel"],
        xaxis_title=None,                # l'axe des dates se suffit à lui-même
        template="plotly_white",          # fond blanc, grille légère
        font=dict(
            family="Inter, system-ui, sans-serif",
            size=13,
            color=config.COLORS["neutral_dark"],
        ),
        legend=dict(
            orientation="h",              # légende horizontale
            yanchor="bottom",
            y=1.02,                       # juste au-dessus du graphique
            xanchor="left",
            x=0,
            font_size=11,
        ),
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode="x unified",           # tooltip unifié au survol
        height=420,
    )

    return fig


def add_forecast_to_figure(
    fig: go.Figure,
    df: pd.DataFrame,
    strings: dict,
) -> go.Figure:
    """
    Ajoute la prévision (fan chart) au graphique de la série temporelle.

    Deux zones visuellement distinctes :
    - validée (≤24 mois) : bleu, trait plein, bande d'IC bleu clair
    - extrapolation (>24 mois) : ambre, tirets, bande d'IC ambre clair

    Une ligne verticale marque la coupure entre les deux zones.

    Paramètres
    ----------
    fig : la figure Plotly de la série temporelle (sortie de make_timeseries_figure)
    df : DataFrame complet (passé à build_scenarios pour l'ajustement Prophet)
    strings : dictionnaire des libellés

    Retourne
    --------
    go.Figure avec la prévision ajoutée
    """
    # ── Générer les scénarios via le module scenarios.py ──
    scenarios = build_scenarios(
    gwsa_mm=df["gwsa_mm"],
    validated_mae_mm=6.1,       # MAE de la CV Prophet (résultat Tâche 10)
)
    forecast_df = scenarios["forecast_df"]

    # Séparer les deux zones
    validated = forecast_df[forecast_df["zone"] == "validated"].copy()
    extrapol = forecast_df[forecast_df["zone"] == "extrapolation"].copy()

    # ── Zone validée (≤24 mois) — bleu ──

    if not validated.empty:
        # Bande d'IC (zone colorée entre yhat_lower et yhat_upper)
        # Technique Plotly : on trace le bord supérieur puis le bord inférieur
        # en sens inverse, et on remplit "toself" pour créer la zone
        fig.add_trace(go.Scatter(
            x=pd.concat([validated["ds"], validated["ds"][::-1]]),
            y=pd.concat([validated["yhat_upper"], validated["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(41, 128, 185, 0.15)",  # bleu transparent
            line=dict(width=0),
            name=strings["forecast_ci"],
            showlegend=True,
            hoverinfo="skip",
        ))

        # Ligne centrale de la prévision validée
        fig.add_trace(go.Scatter(
            x=validated["ds"],
            y=validated["yhat"],
            mode="lines",
            name=strings["forecast_validated"],
            line=dict(color=config.COLORS["validated"], width=2),
            hovertemplate="%{x|%b %Y} : %{y:.1f} mm<extra></extra>",
        ))

    # ── Zone extrapolation (>24 mois) — ambre tirets ──

    if not extrapol.empty:
        # Bande d'IC extrapolation (ambre transparent)
        fig.add_trace(go.Scatter(
            x=pd.concat([extrapol["ds"], extrapol["ds"][::-1]]),
            y=pd.concat([extrapol["yhat_upper"], extrapol["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(230, 126, 34, 0.10)",  # ambre transparent
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Ligne centrale extrapolation (tirets)
        fig.add_trace(go.Scatter(
            x=extrapol["ds"],
            y=extrapol["yhat"],
            mode="lines",
            name=strings["forecast_extrapolation"],
            line=dict(
                color=config.COLORS["extrapolation"],
                width=2,
                dash="dash",
            ),
            hovertemplate="%{x|%b %Y} : %{y:.1f} mm<extra></extra>",
        ))

    # ── Ligne verticale de coupure validé / extrapolation ──
    cutoff_date = scenarios["cutoff_date"]

    fig.add_vline(
        x=cutoff_date,
        line=dict(
            color=config.COLORS["neutral_mid"],
            width=1,
            dash="dashdot",
        ),
        annotation_text=strings["forecast_cutoff"],
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color=config.COLORS["neutral_mid"],
    )

    return fig