# fichier : pipeline/dashboard.py
"""
Fonctions de création des figures Plotly et calcul des KPI
pour le dashboard SASS.

Ce module contient la logique de visualisation.
build_dashboard.py l'orchestre pour produire docs/index.html.
"""

import json

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import zoom as scipy_zoom
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from pipeline import config
from pipeline.trend import mann_kendall_sen, ols_trend_hac, compute_aoi_area_m2, mm_to_km3
from pipeline.indicators import compute_zscore
from pipeline.decomposition import run_stl, run_decomposition_diagnostics
from pipeline.scenarios import build_scenarios, get_scenario_summary, build_multi_scenarios


# ── Chargement des données ────────────────────────────────────

def load_strings(lang: str = "en") -> dict:
    """Charge les libellés d'interface pour la langue donnée."""
    path = config.UI_STRINGS_DIR / f"{lang}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data() -> pd.DataFrame:
    """Charge le cache Parquet principal."""
    return pd.read_parquet(config.SERIES_PARQUET)


def load_freshness(df: pd.DataFrame = None) -> dict:
    """Charge last_refresh.json ou dérive les dates de fraîcheur du DataFrame."""
    path = config.LAST_REFRESH_JSON
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        # Vérifier que les valeurs ne sont pas N/A
        if all(v != "N/A" for v in data.values()):
            return data

    # Dériver les dates depuis le DataFrame si disponible
    if df is not None:
        obs = df[~df["is_imputed"]]
        last_obs = obs.index[-1].strftime("%Y-%m")
        last_all = df.index[-1].strftime("%Y-%m")
        # GRACE et GLDAS ont la même couverture dans notre pipeline
        return {
            "last_grace_month": last_obs,
            "last_gldas_month": last_obs,
            "last_common_month": last_all,
        }

    return {
        "last_grace_month": "N/A",
        "last_gldas_month": "N/A",
        "last_common_month": "N/A",
    }


# ── KPI et données pour le template ──────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Calcule tous les KPI du dashboard à partir du DataFrame."""
    mk_result = mann_kendall_sen(
        series=df["gwsa_mm"],
        is_imputed=df["is_imputed"],
    )
    trend_rate_mm = mk_result["sen_slope_mm_yr"]

    area_m2 = compute_aoi_area_m2(config.AOI_GEOJSON)
    area_km2 = int(area_m2 / 1e6)
    trend_rate_vol = trend_rate_mm * area_m2 / 1e12

    obs = df[~df["is_imputed"]]
    last_anomaly = obs["gwsa_mm"].iloc[-1]
    last_date = obs.index[-1].strftime("%Y-%m")

    zscores = compute_zscore(df["gwsa_mm"], df["is_imputed"])
    last_zscore = zscores.iloc[-1]

    mk_trend = mk_result["mk_trend"]
    mk_pvalue = mk_result["mk_pvalue"]

    # Prévision à 24 mois
    scenarios = build_scenarios(gwsa_mm=df["gwsa_mm"], validated_mae_mm=6.1)
    summary = get_scenario_summary(scenarios)
    row_24 = summary[summary["horizon_months"] == 24]
    forecast_24_abs = row_24.iloc[0]["yhat_mm"] if not row_24.empty else last_anomaly

    # Amplitude saisonnière via STL
    diag = run_decomposition_diagnostics(df["gwsa_mm"])
    seasonal_amplitude = diag["seasonal_amplitude_mm"]

    # Prévision : écart attendu par rapport au dernier mois observé
    forecast_24_delta = forecast_24_abs - last_anomaly

    # Vigilance basée sur le z-score (§6.1)
    if last_zscore < -2:
        vigilance = "high"
    elif last_zscore < -1:
        vigilance = "moderate"
    else:
        vigilance = "low"

    # ── Métriques contextuelles (dérivées, pas de nouveaux calculs) ──
    # Delta vs mois précédent
    prev_anomaly = obs["gwsa_mm"].iloc[-2] if len(obs) >= 2 else last_anomaly
    delta_month = last_anomaly - prev_anomaly

    # Delta vs même mois année précédente
    year_ago_idx = obs.index[-1] - pd.DateOffset(months=12)
    nearest_idx = obs.index.get_indexer([year_ago_idx], method="nearest")[0]
    delta_year = last_anomaly - obs["gwsa_mm"].iloc[nearest_idx]

    # Rang historique
    all_vals = obs["gwsa_mm"].values
    rank = int(np.sum(all_vals <= last_anomaly))
    rank_total = len(all_vals)

    # Interprétation anomalie
    if last_zscore < -2:
        interp_key = "critical"
    elif last_zscore < -1:
        interp_key = "below"
    elif last_zscore < 1:
        interp_key = "near"
    else:
        interp_key = "above"

    # Forecast CI moyen (from scenario data)
    ci_vals = forecast_df = scenarios["forecast_df"]
    validated_rows = ci_vals[ci_vals["zone"] == "validated"]
    if not validated_rows.empty:
        ci_avg = (validated_rows["yhat_upper"] - validated_rows["yhat_lower"]).mean()
    else:
        ci_avg = 0.0

    return {
        # KPI 1 — Dernière anomalie
        "kpi_anomaly": f"{last_anomaly:+.1f}",
        "kpi_anomaly_raw": last_anomaly,
        "kpi_zscore": f"{last_zscore:+.1f}",
        "kpi_last_date": last_date,
        # KPI 2 — Tendance (Sen) + volume
        "kpi_trend_rate": f"{trend_rate_mm:+.1f}",
        "kpi_trend_vol": f"{trend_rate_vol:+.2f}",
        "area_km2": f"{area_km2:,}",
        # KPI 3 — Prévision 24 mois
        "kpi_forecast_24": f"{forecast_24_abs:+.1f}",
        "kpi_forecast_24_delta": f"{forecast_24_delta:+.1f}",
        # KPI 4 — Vigilance
        "kpi_vigilance": vigilance,
        # Technique (disponible pour le template)
        "kpi_mk_trend": mk_trend,
        "kpi_mk_pvalue": f"< 0.001" if mk_pvalue < 0.001 else f"= {mk_pvalue:.3f}",
        "seasonal_amplitude_mm": f"{seasonal_amplitude:.1f}",
        # ── Contextuels (nouveaux) ──
        "kpi_delta_month": f"{delta_month:+.1f}",
        "kpi_delta_year": f"{delta_year:+.1f}",
        "kpi_rank": rank,
        "kpi_rank_total": rank_total,
        "kpi_interp_key": interp_key,
        "kpi_forecast_mae": "6.1",
        "kpi_forecast_ci_avg": f"{ci_avg:.0f}",
    }


def make_sparkline_data(df: pd.DataFrame) -> dict:
    """Génère les données de mini-sparklines pour les KPI cards (derniers 36 mois)."""
    obs = df[~df["is_imputed"]].copy()
    recent = obs.tail(36)

    gwsa_vals = recent["gwsa_mm"].tolist()

    mk = mann_kendall_sen(df["gwsa_mm"], df["is_imputed"])
    slope = mk["sen_slope_mm_yr"]
    x_months = list(range(len(recent)))
    trend_vals = [slope * (m / 12) for m in x_months]

    return {
        "gwsa_sparkline": gwsa_vals,
        "trend_sparkline": trend_vals,
        "dates_sparkline": [d.strftime("%Y-%m") for d in recent.index],
    }


# ── Figures Plotly ────────────────────────────────────────────

def make_timeseries_figure(df: pd.DataFrame, strings: dict) -> go.Figure:
    """Graphique principal : série gwsa_mm + incertitude + tendance OLS + fan chart."""
    fig = go.Figure()

    # Zone grisée lacune 2017-2018
    fig.add_vrect(
        x0="2017-06-01", x1="2018-06-01",
        fillcolor=config.COLORS["gap_zone"], opacity=0.9,
        layer="below", line_width=0,
        annotation_text=strings["ts_gap_label"],
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color=config.COLORS["text_light"],
    )

    # Bande d'incertitude (± uncertainty_mm) sur les mois observés
    obs = df[~df["is_imputed"]].copy()
    if "uncertainty_mm" in df.columns:
        obs_unc = obs.copy()
        upper = obs_unc["gwsa_mm"] + obs_unc["uncertainty_mm"]
        lower = obs_unc["gwsa_mm"] - obs_unc["uncertainty_mm"]
        fig.add_trace(go.Scatter(
            x=pd.concat([obs_unc.index.to_series(), obs_unc.index.to_series()[::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself",
            fillcolor=config.COLORS["primary_bg"],
            line=dict(width=0),
            name=strings.get("ts_uncertainty", "Uncertainty"),
            showlegend=True,
            hoverinfo="skip",
        ))

    # Couche basse : série complète en pointillés (mois imputés)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["gwsa_mm"],
        mode="lines",
        name=strings["ts_imputed"],
        line=dict(color=config.COLORS["imputed"], width=1.2, dash="dot"),
        hoverinfo="skip", showlegend=True,
    ))

    # Couche haute : mois observés en trait plein
    fig.add_trace(go.Scatter(
        x=obs.index, y=obs["gwsa_mm"],
        mode="lines",
        name=strings["ts_observed"],
        line=dict(color=config.COLORS["primary"], width=2),
        hovertemplate="%{x|%b %Y} : %{y:.1f} mm<extra></extra>",
    ))

    # Ligne de tendance OLS (tirets violets)
    ols = ols_trend_hac(df["gwsa_mm"], df["is_imputed"])
    t0 = obs.index[0]
    x_years_end = (obs.index[-1] - t0).days / 365.25
    y_start = ols["intercept"]
    y_end = ols["intercept"] + ols["slope_mm_yr"] * x_years_end
    fig.add_trace(go.Scatter(
        x=[obs.index[0], obs.index[-1]],
        y=[y_start, y_end],
        mode="lines",
        name=strings["ts_trend_line"],
        line=dict(color=config.COLORS["trend_line"], width=2, dash="dash"),
        hovertemplate="Trend: %{y:.1f} mm<extra></extra>",
    ))

    # Fan chart prévision
    _add_forecast(fig, df, strings)

    # ── Annotations missions GRACE / GRACE-FO ──
    fig.add_annotation(
        x="2003-06-01", y=1, yref="paper", yanchor="bottom",
        text="GRACE", showarrow=False,
        font=dict(size=10, color=config.COLORS["text_light"]),
        bgcolor="rgba(255,255,255,0.7)",
    )
    fig.add_annotation(
        x="2019-06-01", y=1, yref="paper", yanchor="bottom",
        text="GRACE-FO", showarrow=False,
        font=dict(size=10, color=config.COLORS["text_light"]),
        bgcolor="rgba(255,255,255,0.7)",
    )

    fig.update_layout(
        yaxis_title=strings["ts_ylabel"],
        xaxis_title=None,
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13,
                  color=config.COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font_size=11),
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
        height=560,
    )
    return fig


def _add_forecast(fig: go.Figure, df: pd.DataFrame, strings: dict):
    """Ajoute le fan chart (validé + extrapolation) au graphique."""
    scenarios = build_scenarios(
        gwsa_mm=df["gwsa_mm"],
        validated_mae_mm=6.1,
    )
    forecast_df = scenarios["forecast_df"]

    validated = forecast_df[forecast_df["zone"] == "validated"].copy()
    extrapol = forecast_df[forecast_df["zone"] == "extrapolation"].copy()

    if not validated.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([validated["ds"], validated["ds"][::-1]]),
            y=pd.concat([validated["yhat_upper"], validated["yhat_lower"][::-1]]),
            fill="toself", fillcolor=config.COLORS["primary_bg"],
            line=dict(width=0), name=strings["forecast_ci"],
            showlegend=True, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=validated["ds"], y=validated["yhat"],
            mode="lines", name=strings["forecast_validated"],
            line=dict(color=config.COLORS["primary_light"], width=2.5),
            hovertemplate="%{x|%b %Y} : %{y:.1f} mm<extra></extra>",
        ))

    if not extrapol.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([extrapol["ds"], extrapol["ds"][::-1]]),
            y=pd.concat([extrapol["yhat_upper"], extrapol["yhat_lower"][::-1]]),
            fill="toself", fillcolor=config.COLORS["warn_light"],
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=extrapol["ds"], y=extrapol["yhat"],
            mode="lines", name=strings["forecast_extrapolation"],
            line=dict(color=config.COLORS["extrapolation"], width=2, dash="dash"),
            hovertemplate="%{x|%b %Y} : %{y:.1f} mm<extra></extra>",
        ))

    fig.add_vline(
        x=scenarios["cutoff_date"],
        line=dict(color=config.COLORS["text_light"], width=1, dash="dashdot"),
        annotation_text=strings["forecast_cutoff"],
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color=config.COLORS["text_mid"],
    )


def make_stl_figure(df: pd.DataFrame, strings: dict) -> go.Figure:
    """Graphique STL : 3 sous-graphiques empilés (tendance, saisonnalité, résidu)."""
    stl_df = run_stl(df["gwsa_mm"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            strings["stl_trend_label"],
            strings["stl_seasonal_label"],
            strings["stl_resid_label"],
        ),
    )

    # Tendance — violet épais
    fig.add_trace(go.Scatter(
        x=stl_df.index, y=stl_df["trend"],
        mode="lines", line=dict(color=config.COLORS["trend_line"], width=2.5),
        name=strings["stl_trend_label"], showlegend=False,
        hovertemplate="%{x|%b %Y} : %{y:.1f} mm<extra></extra>",
    ), row=1, col=1)

    # Saisonnalité — orange
    fig.add_trace(go.Scatter(
        x=stl_df.index, y=stl_df["seasonal"],
        mode="lines", line=dict(color=config.COLORS["seasonal"], width=1.5),
        name=strings["stl_seasonal_label"], showlegend=False,
        hovertemplate="%{x|%b %Y} : %{y:.1f} mm<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=config.COLORS["text_light"],
                  line_width=0.8, row=2, col=1)

    # Résidu — gris
    fig.add_trace(go.Scatter(
        x=stl_df.index, y=stl_df["resid"],
        mode="lines", line=dict(color=config.COLORS["residual"], width=1),
        name=strings["stl_resid_label"], showlegend=False,
        hovertemplate="%{x|%b %Y} : %{y:.1f} mm<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=config.COLORS["text_light"],
                  line_width=0.8, row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12,
                  color=config.COLORS["text"]),
        margin=dict(l=60, r=20, t=30, b=30),
        height=500, showlegend=False,
        hovermode="x unified",
    )
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=config.COLORS["text_mid"])

    return fig


def make_gldas_contribution_figure(df: pd.DataFrame, strings: dict) -> go.Figure:
    """Bar chart horizontal : contribution relative TWSA, GLDAS, GWSA."""
    obs = df[~df["is_imputed"]].copy()

    # Écart-type de chaque composante (variabilité = "poids" du signal)
    twsa_std = (obs["twsa_cm"] * 10).std()   # cm → mm
    gldas_std = obs["gldas_anom_mm"].std()
    gwsa_std = obs["gwsa_mm"].std()

    labels = [
        strings.get("gldas_bar_twsa", "TWSA (GRACE)"),
        strings.get("gldas_bar_gldas", "Surface (GLDAS)"),
        strings.get("gldas_bar_gwsa", "GWSA (proxy)"),
    ]
    values = [twsa_std, gldas_std, gwsa_std]
    colors = [
        config.COLORS["primary_light"],
        config.COLORS["seasonal"],
        config.COLORS["primary"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f} mm" for v in values],
        textposition="outside",
        hovertemplate="%{y} : %{x:.1f} mm (σ)<extra></extra>",
    ))

    # Pourcentage GLDAS / TWSA
    pct_gldas = (gldas_std / twsa_std * 100) if twsa_std > 0 else 0
    fig.add_annotation(
        x=max(values) * 0.95,
        y=1,
        text=f"{pct_gldas:.0f}% of TWSA",
        showarrow=False,
        font=dict(size=11, color=config.COLORS["text_mid"]),
        xanchor="right",
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13,
                  color=config.COLORS["text"]),
        margin=dict(l=120, r=60, t=20, b=30),
        height=220,
        xaxis_title=strings.get("gldas_bar_xaxis", "Variability σ (mm)"),
        showlegend=False,
        hovermode="y unified",
    )
    return fig


def make_aoi_map(strings: dict) -> go.Figure:
    """Carte géographique : emprise SASS + grille mascon + villes repères."""
    fig = go.Figure()

    aoi = gpd.read_file(config.AOI_GEOJSON)
    poly = aoi.geometry.iloc[0]

    # ── Polygone SASS (remplissage léger) ──
    coords = list(poly.exterior.coords)
    lons_aoi = [c[0] for c in coords]
    lats_aoi = [c[1] for c in coords]

    fig.add_trace(go.Scattergeo(
        lon=lons_aoi, lat=lats_aoi,
        mode="lines",
        line=dict(width=3, color="#3D4F2F"),
        fill="toself",
        fillcolor="rgba(163, 177, 138, 0.25)",
        name="SASS / NWSAS",
        hoverinfo="name",
    ))

    # ── Grille mascon 0.5° (si NetCDF disponible) ──
    nc_path = config.MASCON_NC_PATH
    if nc_path.exists():
        ds = xr.open_dataset(nc_path)
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
        mascon_ids = ds["mascon_ID"].sel(
            lat=slice(config.BBOX_LAT_MIN, config.BBOX_LAT_MAX),
            lon=slice(config.BBOX_LON_MIN, config.BBOX_LON_MAX),
        )
        lats = mascon_ids.lat.values
        lons = mascon_ids.lon.values

        # Dessiner les lignes de grille horizontales
        for lat_val in lats:
            fig.add_trace(go.Scattergeo(
                lon=[lons[0] - 0.25, lons[-1] + 0.25],
                lat=[lat_val - 0.25, lat_val - 0.25],
                mode="lines",
                line=dict(width=0.5, color="rgba(0,0,0,0.12)"),
                hoverinfo="skip", showlegend=False,
            ))
        # Dessiner les lignes de grille verticales
        for lon_val in lons:
            fig.add_trace(go.Scattergeo(
                lon=[lon_val - 0.25, lon_val - 0.25],
                lat=[lats[0] - 0.25, lats[-1] + 0.25],
                mode="lines",
                line=dict(width=0.5, color="rgba(0,0,0,0.12)"),
                hoverinfo="skip", showlegend=False,
            ))
        ds.close()

    # ── Villes repères ──
    cities = [
        {"name": "Ouargla", "lon": 5.33, "lat": 31.95},
        {"name": "Ghardaïa", "lon": 3.67, "lat": 32.49},
        {"name": "Tozeur", "lon": 8.13, "lat": 33.92},
        {"name": "Ghadames", "lon": 9.50, "lat": 30.13},
        {"name": "In Salah", "lon": 2.47, "lat": 27.19},
    ]
    fig.add_trace(go.Scattergeo(
        lon=[c["lon"] for c in cities],
        lat=[c["lat"] for c in cities],
        mode="markers+text",
        marker=dict(size=7, color="#0F172A", symbol="circle",
                    line=dict(width=1.5, color="white")),
        text=[c["name"] for c in cities],
        textposition="top center",
        textfont=dict(size=10, color="#0F172A", family="Inter, sans-serif"),
        name="Cities",
        hovertemplate="%{text}<br>%{lon:.2f}°E, %{lat:.2f}°N<extra></extra>",
    ))

    fig.update_geos(
        fitbounds="locations",
        showland=True, landcolor="#F5F5F0",
        showocean=True, oceancolor="#E8F4FD",
        showlakes=False,
        showcountries=True, countrycolor="#999999",
        countrywidth=1,
        showcoastlines=True, coastlinecolor="#AAAAAA",
        projection_type="natural earth",
        lonaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.08)",
                     dtick=5),
        lataxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.08)",
                     dtick=5),
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=500,
        showlegend=False,
        font=dict(family="Inter, system-ui, sans-serif", size=12,
                  color=config.COLORS["text"]),
    )

    return fig


# ── Nouvelles figures (refonte dashboard) ─────────────────────

def make_annual_bar_figure(df: pd.DataFrame, strings: dict) -> go.Figure:
    """Bar chart annuel — une barre par année, gwsa_mm moyen observé."""
    obs = df[~df["is_imputed"]].copy()
    annual = obs.groupby(obs.index.year)["gwsa_mm"].agg(["mean", "count"])

    # Gradient bleu (valeurs hautes) → rouge (valeurs basses)
    vals = annual["mean"].values
    vmin, vmax = vals.min(), vals.max()
    span = vmax - vmin if vmax != vmin else 1.0

    def _gradient_color(v):
        """Interpolate from alert (red, low) to primary_light (blue, high)."""
        t = (v - vmin) / span  # 0 = min (red), 1 = max (blue)
        r_lo, g_lo, b_lo = 196, 69, 54    # alert #C44536
        r_hi, g_hi, b_hi = 61, 124, 140   # primary_light #3D7C8C
        r = int(r_lo + (r_hi - r_lo) * t)
        g = int(g_lo + (g_hi - g_lo) * t)
        b = int(b_lo + (b_hi - b_lo) * t)
        return f"rgb({r},{g},{b})"

    # Années partielles (< 10 mois) : même gradient mais plus transparent
    full_year_threshold = 10
    colors = []
    for v, c in zip(vals, annual["count"]):
        base = _gradient_color(v)
        if c < full_year_threshold:
            # Convert to rgba with reduced opacity
            rgb = base[4:-1]  # strip "rgb(" and ")"
            colors.append(f"rgba({rgb},0.45)")
        else:
            colors.append(base)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=annual.index.astype(str),
        y=annual["mean"],
        marker_color=colors,
        hovertemplate="%{x}: %{y:.1f} mm<extra></extra>",
    ))

    fig.update_layout(
        yaxis_title=strings.get("annual_bar_ylabel", "Annual mean anomaly (mm)"),
        xaxis_title=None,
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13,
                  color=config.COLORS["text"]),
        margin=dict(l=60, r=20, t=10, b=40),
        height=260,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(tickangle=-45, dtick=2)
    return fig


def make_seasonal_bar_figure(df: pd.DataFrame, strings: dict) -> go.Figure:
    """Bar chart 12 mois (Jan–Déc) — composante saisonnière STL moyenne."""
    stl_df = run_stl(df["gwsa_mm"])
    monthly_seasonal = stl_df.groupby(stl_df.index.month)["seasonal"].mean()

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=month_labels,
        y=monthly_seasonal.values,
        marker_color=config.COLORS["seasonal"],
        hovertemplate="%{x}: %{y:.2f} mm<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot",
                  line_color=config.COLORS["text_light"], line_width=0.8)

    fig.update_layout(
        yaxis_title=strings.get("seasonal_bar_ylabel", "Seasonal deviation (mm)"),
        xaxis_title=None,
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13,
                  color=config.COLORS["text"]),
        margin=dict(l=60, r=20, t=10, b=30),
        height=280,
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def make_scenario_figure(df: pd.DataFrame, strings: dict) -> go.Figure:
    """Graphique dédié prévision : 5 ans historique + 5 ans prévision avec IC."""
    scenarios = build_scenarios(gwsa_mm=df["gwsa_mm"], validated_mae_mm=6.1)
    forecast_df = scenarios["forecast_df"]
    last_obs_date = scenarios["last_obs_date"]
    cutoff_date = scenarios["cutoff_date"]

    # 5 dernières années d'observations
    obs_start = last_obs_date - pd.DateOffset(years=5)
    obs = df[~df["is_imputed"]].copy()
    obs_recent = obs.loc[obs_start:]

    validated = forecast_df[forecast_df["zone"] == "validated"].copy()
    extrapol = forecast_df[forecast_df["zone"] == "extrapolation"].copy()

    fig = go.Figure()

    # Observations récentes
    fig.add_trace(go.Scatter(
        x=obs_recent.index, y=obs_recent["gwsa_mm"],
        mode="lines",
        name=strings.get("ts_observed", "Observed"),
        line=dict(color=config.COLORS["primary"], width=2),
        hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra></extra>",
    ))

    # Bande IC validée
    if not validated.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([validated["ds"], validated["ds"][::-1]]),
            y=pd.concat([validated["yhat_upper"], validated["yhat_lower"][::-1]]),
            fill="toself", fillcolor=config.COLORS["primary_bg"],
            line=dict(width=0),
            name=strings.get("forecast_ci", "95% CI"),
            showlegend=True, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=validated["ds"], y=validated["yhat"],
            mode="lines",
            name=strings.get("forecast_validated", "Validated forecast"),
            line=dict(color=config.COLORS["primary_light"], width=2.5),
            hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra></extra>",
        ))

    # Bande IC extrapolation
    if not extrapol.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([extrapol["ds"], extrapol["ds"][::-1]]),
            y=pd.concat([extrapol["yhat_upper"], extrapol["yhat_lower"][::-1]]),
            fill="toself", fillcolor=config.COLORS["warn_light"],
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=extrapol["ds"], y=extrapol["yhat"],
            mode="lines",
            name=strings.get("forecast_extrapolation", "Extrapolation"),
            line=dict(color=config.COLORS["extrapolation"], width=2, dash="dash"),
            hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra></extra>",
        ))

    # Ligne de séparation
    fig.add_vline(
        x=cutoff_date,
        line=dict(color=config.COLORS["text_light"], width=1, dash="dashdot"),
        annotation_text=strings.get("forecast_cutoff", "Validated limit"),
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color=config.COLORS["text_mid"],
    )

    fig.update_layout(
        yaxis_title=strings.get("scenario_ylabel", "GWSA anomaly (mm)"),
        xaxis_title=None,
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13,
                  color=config.COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font_size=11),
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode="x unified",
        height=380,
    )
    return fig


# ── Tableaux de données pour le template ──────────────────────

def compute_trend_table(df: pd.DataFrame) -> dict:
    """Calcule les détails OLS+HAC et Sen pour le tableau comparatif."""
    ols = ols_trend_hac(df["gwsa_mm"], df["is_imputed"])
    mk_sen = mann_kendall_sen(df["gwsa_mm"], df["is_imputed"])
    area_m2 = compute_aoi_area_m2(config.AOI_GEOJSON)

    return {
        "ols_slope_mm": f"{ols['slope_mm_yr']:+.2f}",
        "ols_ci": f"[{ols['ci_lower_mm_yr']:+.2f}, {ols['ci_upper_mm_yr']:+.2f}]",
        "ols_pvalue": f"< 0.001" if ols["pvalue"] < 0.001 else f"{ols['pvalue']:.3f}",
        "ols_slope_km3": f"{mm_to_km3(ols['slope_mm_yr'], area_m2):+.2f}",
        "sen_slope_mm": f"{mk_sen['sen_slope_mm_yr']:+.2f}",
        "sen_ci": "\u2014",
        "sen_pvalue": f"< 0.001" if mk_sen["mk_pvalue"] < 0.001 else f"{mk_sen['mk_pvalue']:.3f}",
        "sen_slope_km3": f"{mm_to_km3(mk_sen['sen_slope_mm_yr'], area_m2):+.2f}",
    }


def compute_forecast_milestones(df: pd.DataFrame) -> list[dict]:
    """Calcule le tableau des jalons de prévision (12, 24, 36, 48, 60 mois)."""
    scenarios = build_scenarios(gwsa_mm=df["gwsa_mm"], validated_mae_mm=6.1)
    summary = get_scenario_summary(scenarios)

    milestones = []
    for _, row in summary.iterrows():
        milestones.append({
            "horizon": int(row["horizon_months"]),
            "date": row["date"].strftime("%Y-%m"),
            "yhat": f"{row['yhat_mm']:+.1f}",
            "ci": f"[{row['lower_mm']:+.1f}, {row['upper_mm']:+.1f}]",
            "zone": row["zone"],
        })
    return milestones


# ── Nouvelles figures (refonte layout) ──────────────────────────

def make_decision_mini_bar(df: pd.DataFrame, strings: dict) -> go.Figure:
    """Mini bar chart : niveaux annuels moyens projetés (~7 barres, panneau décideur)."""
    scenarios = build_scenarios(gwsa_mm=df["gwsa_mm"], validated_mae_mm=6.1)
    forecast_df = scenarios["forecast_df"]
    last_obs_date = scenarios["last_obs_date"]

    # Années observées récentes (2 dernières)
    obs = df[~df["is_imputed"]].copy()
    obs_annual = obs.groupby(obs.index.year)["gwsa_mm"].mean()
    recent = obs_annual.tail(2)

    # Années projetées (moyenne annuelle de yhat)
    fc_annual = forecast_df.groupby(forecast_df["ds"].dt.year)["yhat"].mean()

    # Combiner, garder les observations quand les deux existent
    combined = pd.concat([recent, fc_annual])
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.tail(7)

    vals = combined.values
    vmin, vmax = vals.min(), vals.max()
    span = vmax - vmin if vmax != vmin else 1.0

    # Gradient alert (#B7410E) → primary_light (#4A90A4)
    colors = []
    for v in vals:
        t = (v - vmin) / span
        r = int(183 + (74 - 183) * t)
        g = int(65 + (144 - 65) * t)
        b = int(14 + (164 - 14) * t)
        colors.append(f"rgb({r},{g},{b})")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(y) for y in combined.index],
        y=combined.values,
        marker_color=colors,
        text=[f"{v:.0f}" for v in combined.values],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="%{x}: %{y:.1f} mm<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=11,
                  color=config.COLORS["text"]),
        margin=dict(l=35, r=10, t=10, b=25),
        height=180,
        showlegend=False,
        yaxis_title=strings.get("ts_ylabel", "mm"),
        yaxis=dict(title_font_size=9),
        xaxis=dict(tickfont_size=9),
    )
    return fig


def make_multi_scenario_figure(df: pd.DataFrame, strings: dict) -> go.Figure:
    """Graphique 3 scénarios : central + sécheresse + gestion renforcée."""
    multi = build_multi_scenarios(
        gwsa_mm=df["gwsa_mm"],
        is_imputed=df["is_imputed"],
        validated_mae_mm=6.1,
    )
    central_df = multi["central_df"]
    dry_df = multi["dry_df"]
    wet_df = multi["wet_df"]
    last_obs_date = multi["last_obs_date"]
    cutoff_date = multi["cutoff_date"]

    # 5 dernières années d'observations
    obs_start = last_obs_date - pd.DateOffset(years=5)
    obs = df[~df["is_imputed"]].copy()
    obs_recent = obs.loc[obs_start:]

    fig = go.Figure()

    # Observations récentes
    fig.add_trace(go.Scatter(
        x=obs_recent.index, y=obs_recent["gwsa_mm"],
        mode="lines",
        name=strings.get("ts_observed", "Observed"),
        line=dict(color=config.COLORS["primary"], width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra></extra>",
    ))

    # Bande IC scénario central
    fig.add_trace(go.Scatter(
        x=pd.concat([central_df["ds"], central_df["ds"][::-1]]),
        y=pd.concat([central_df["yhat_upper"], central_df["yhat_lower"][::-1]]),
        fill="toself", fillcolor=config.COLORS["primary_bg"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Ligne centrale
    fig.add_trace(go.Scatter(
        x=central_df["ds"], y=central_df["yhat"],
        mode="lines",
        name=strings.get("scenario_central", "Trend continuation"),
        line=dict(color=config.COLORS["primary_light"], width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra></extra>",
    ))

    # Scénario sécheresse
    fig.add_trace(go.Scatter(
        x=dry_df["ds"], y=dry_df["yhat"],
        mode="lines",
        name=strings.get("scenario_dry_label", "Dry years"),
        line=dict(color=config.COLORS["scenario_dry"], width=2, dash="dash"),
        hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra></extra>",
    ))

    # Scénario gestion renforcée
    fig.add_trace(go.Scatter(
        x=wet_df["ds"], y=wet_df["yhat"],
        mode="lines",
        name=strings.get("scenario_wet_label", "Enhanced management"),
        line=dict(color=config.COLORS["scenario_wet"], width=2, dash="dash"),
        hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra></extra>",
    ))

    # Ligne de séparation horizon validé
    fig.add_vline(
        x=cutoff_date,
        line=dict(color=config.COLORS["text_light"], width=1, dash="dashdot"),
        annotation_text=strings.get("forecast_cutoff", "Validated limit"),
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color=config.COLORS["text_mid"],
    )

    fig.update_layout(
        yaxis_title=strings.get("scenario_ylabel", "GWSA anomaly (mm)"),
        xaxis_title=None,
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13,
                  color=config.COLORS["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font_size=11),
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode="x unified",
        height=380,
    )
    return fig
