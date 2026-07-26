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
    """Charge le cache Parquet principal et applique le prétraitement."""
    path = config.SERIES_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"Cache Parquet introuvable : {path}\n"
            "Lancez d'abord le pipeline d'ingestion (uv run python -m pipeline.ingest)."
        )
    df = pd.read_parquet(path)

    # Tronquer au premier mois où gwsa_mm existe (GRACE démarre avril 2002)
    first_valid = df["gwsa_mm"].first_valid_index()
    if first_valid is not None:
        df = df.loc[first_valid:]

    # Appliquer le prétraitement si is_imputed n'existe pas encore
    if "is_imputed" not in df.columns:
        from pipeline.preprocessing import reindex_monthly, interpolate_gaps
        df = reindex_monthly(df)
        df = interpolate_gaps(df)

    return df

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
    scenarios = build_scenarios(gwsa_mm=df["gwsa_mm"], validated_mae_mm=config.VALIDATED_MAE_MM)
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
    ci_vals = scenarios["forecast_df"]
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
        "kpi_mk_pvalue": "< 0.001" if mk_pvalue < 0.001 else f"= {mk_pvalue:.3f}",
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
        x0=config.GAP_START, x1=config.GAP_END,
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
            showlegend=False,
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
        showlegend=False,
    ))

    # Fan chart prévision
    _add_forecast(fig, df, strings)

    # ── Annotations missions GRACE / GRACE-FO ──
    fig.add_annotation(
        x=config.GRACE_MISSION_LABEL_DATE, y=1, yref="paper", yanchor="bottom",
        text="GRACE", showarrow=False,
        font=dict(size=10, color=config.COLORS["text_light"]),
        bgcolor="rgba(255,255,255,0.7)",
    )
    fig.add_annotation(
        x=config.GRACEFO_MISSION_LABEL_DATE, y=1, yref="paper", yanchor="bottom",
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
                    xanchor="center", x=0.5, font_size=11,
                    entrywidth=0.22, entrywidthmode="fraction"),
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
        height=560,
    )
    return fig


def _add_forecast(fig: go.Figure, df: pd.DataFrame, strings: dict):
    """Ajoute le fan chart (validé + extrapolation) au graphique."""
    scenarios = build_scenarios(
        gwsa_mm=df["gwsa_mm"],
        validated_mae_mm=config.VALIDATED_MAE_MM,
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
            showlegend=False, hoverinfo="skip",
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
    """Signal decomposition: TWSA = GWSA (proxy) + GLDAS surface stores.

    Two panels:
      Top — time series overlay of TWSA, GLDAS, GWSA to show relative magnitudes
      Bottom — horizontal bar with variability (σ) and % contribution
    """
    obs = df[~df["is_imputed"]].copy()

    twsa_mm = obs["twsa_cm"] * 10  # cm → mm
    gldas_mm = obs["gldas_anom_mm"]
    gwsa_mm = obs["gwsa_mm"]

    twsa_std = twsa_mm.std()
    gldas_std = gldas_mm.std()
    gwsa_std = gwsa_mm.std()
    pct_gldas = (gldas_std / twsa_std * 100) if twsa_std > 0 else 0
    pct_gwsa = (gwsa_std / twsa_std * 100) if twsa_std > 0 else 0

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.15,
        subplot_titles=(
            "TWSA vs GWSA vs GLDAS (mm)",
            f"Variability σ (mm) — GLDAS = {pct_gldas:.0f}% of TWSA",
        ),
    )

    # ── Top panel: time series overlay ──
    fig.add_trace(go.Scatter(
        x=obs.index, y=twsa_mm,
        mode="lines", name=strings.get("gldas_bar_twsa", "TWSA (GRACE)"),
        line=dict(color=config.COLORS["primary_light"], width=1.8),
        hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra>TWSA</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=obs.index, y=gwsa_mm,
        mode="lines", name=strings.get("gldas_bar_gwsa", "GWSA (proxy)"),
        line=dict(color=config.COLORS["primary"], width=2),
        hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra>GWSA</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=obs.index, y=gldas_mm,
        mode="lines", name=strings.get("gldas_bar_gldas", "Surface (GLDAS)"),
        line=dict(color=config.COLORS["seasonal"], width=1.5, dash="dot"),
        hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra>GLDAS</extra>",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color=config.COLORS["text_light"],
                  line_width=0.6, row=1, col=1)

    # ── Bottom panel: horizontal bars with σ ──
    bar_labels = [
        strings.get("gldas_bar_twsa", "TWSA (GRACE)"),
        strings.get("gldas_bar_gwsa", "GWSA (proxy)"),
        strings.get("gldas_bar_gldas", "Surface (GLDAS)"),
    ]
    bar_values = [twsa_std, gwsa_std, gldas_std]
    bar_colors = [
        config.COLORS["primary_light"],
        config.COLORS["primary"],
        config.COLORS["seasonal"],
    ]
    bar_pcts = [100.0, pct_gwsa, pct_gldas]

    fig.add_trace(go.Bar(
        y=bar_labels, x=bar_values,
        orientation="h", marker_color=bar_colors,
        text=[f"σ = {v:.1f} mm ({p:.0f}%)" for v, p in zip(bar_values, bar_pcts)],
        textposition="outside",
        hovertemplate="%{y}: σ = %{x:.1f} mm<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12,
                  color=config.COLORS["text"]),
        margin=dict(l=60, r=80, t=30, b=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font_size=10),
        hovermode="x unified",
    )
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font = dict(size=10, color=config.COLORS["text_mid"])

    fig.update_yaxes(title_text="mm", row=1, col=1, title_font_size=10)
    fig.update_xaxes(row=2, col=1, range=[0, twsa_std * 1.4])

    return fig


def _get_mascon_cells(poly) -> list[dict]:
    """Identify real ~3° mascon cells that intersect the SASS polygon.

    JPL mascons are ~3 arc-degree equal-area cells.  The NetCDF stores
    data on a 0.5° grid, but adjacent 0.5° pixels sharing the same
    mascon_ID belong to the same physical mascon.  This function groups
    them and returns bounding boxes of each real mascon cell.
    """
    from shapely.geometry import box as shapely_box

    nc_path = config.MASCON_NC_PATH
    if not nc_path.exists():
        return []

    ds = xr.open_dataset(nc_path)
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    mascon_ids = ds["mascon_ID"].sel(
        lat=slice(config.BBOX_LAT_MIN, config.BBOX_LAT_MAX),
        lon=slice(config.BBOX_LON_MIN, config.BBOX_LON_MAX),
    )
    lats = mascon_ids.lat.values
    lons = mascon_ids.lon.values
    ids = mascon_ids.values
    ds.close()

    unique_ids = np.unique(ids[~np.isnan(ids)])
    cells = []
    for mid in unique_ids:
        mask = ids == mid
        lat_idx, lon_idx = np.where(mask)
        lat_min = lats[lat_idx].min() - 0.25
        lat_max = lats[lat_idx].max() + 0.25
        lon_min = lons[lon_idx].min() - 0.25
        lon_max = lons[lon_idx].max() + 0.25

        mascon_box = shapely_box(lon_min, lat_min, lon_max, lat_max)
        if mascon_box.intersects(poly):
            overlap = mascon_box.intersection(poly).area / mascon_box.area
            if overlap > 0.05:  # skip barely touching cells
                cells.append({
                    "id": int(mid),
                    "lon_min": lon_min, "lon_max": lon_max,
                    "lat_min": lat_min, "lat_max": lat_max,
                    "overlap": overlap,
                })
    return cells


def make_aoi_map(strings: dict) -> go.Figure:
    """Carte cartographique professionnelle : SASS + mascons réels (~3°).

    Affiche les véritables cellules mascon JPL (~3 arc-degrés) qui
    intersectent le polygone SASS, sur fond Natural Earth.
    """
    fig = go.Figure()

    aoi = gpd.read_file(config.AOI_GEOJSON)
    poly = aoi.geometry.iloc[0]

    # ── 1. Real mascon cells (~3° each) — dashed grey outlines ──
    mascon_cells = _get_mascon_cells(poly)
    for i, cell in enumerate(mascon_cells):
        is_first = (i == 0)
        rect_lons = [cell["lon_min"], cell["lon_max"], cell["lon_max"],
                     cell["lon_min"], cell["lon_min"]]
        rect_lats = [cell["lat_min"], cell["lat_min"], cell["lat_max"],
                     cell["lat_max"], cell["lat_min"]]

        fig.add_trace(go.Scattergeo(
            lon=rect_lons, lat=rect_lats,
            mode="lines",
            line=dict(width=1, color=config.COLORS["text_light"], dash="dash"),
            name=strings.get("map_mascon_legend", "GRACE measurement resolution (~300 km)") if is_first else "",
            legendgroup="mascon",
            showlegend=is_first,
            hovertemplate=(
                f"Mascon #{cell['id']}<br>"
                f"{cell['lon_min']:.1f}–{cell['lon_max']:.1f}°E, "
                f"{cell['lat_min']:.1f}–{cell['lat_max']:.1f}°N<br>"
                f"SASS overlap: {cell['overlap']*100:.0f}%"
                "<extra></extra>"
            ),
        ))

    # ── 2. SASS boundary polygon ──
    coords = list(poly.exterior.coords)
    lons_aoi = [c[0] for c in coords]
    lats_aoi = [c[1] for c in coords]

    fig.add_trace(go.Scattergeo(
        lon=lons_aoi, lat=lats_aoi,
        mode="lines",
        line=dict(width=2.8, color=config.COLORS["primary"]),
        fill="toself",
        fillcolor=config.COLORS["primary_bg"],
        name="SASS / NWSAS",
        hovertemplate="SASS<br>%{lon:.2f}°E, %{lat:.2f}°N<extra></extra>",
    ))

    # ── 3. Cities ──
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
        marker=dict(
            size=6, color=config.COLORS["text"], symbol="circle",
            line=dict(width=1.2, color=config.COLORS["bg_card"]),
        ),
        text=[c["name"] for c in cities],
        textposition="top center",
        textfont=dict(size=9, color=config.COLORS["text"], family="Inter, sans-serif"),
        name=strings.get("map_cities", "Cities"),
        hovertemplate="%{text}<br>%{lon:.2f}°E, %{lat:.2f}°N<extra></extra>",
    ))

    # ── 4. Geographic base map (Natural Earth) ──
    center_lon = (config.BBOX_LON_MIN + config.BBOX_LON_MAX) / 2
    center_lat = (config.BBOX_LAT_MIN + config.BBOX_LAT_MAX) / 2

    fig.update_geos(
        projection_type="mercator",
        center=dict(lon=center_lon, lat=center_lat),
        lonaxis=dict(
            range=[config.BBOX_LON_MIN - 2, config.BBOX_LON_MAX + 2],
            showgrid=True, gridwidth=0.5,
            gridcolor=config.COLORS["border"], dtick=5,
        ),
        lataxis=dict(
            range=[config.BBOX_LAT_MIN - 2, config.BBOX_LAT_MAX + 2],
            showgrid=True, gridwidth=0.5,
            gridcolor=config.COLORS["border"], dtick=5,
        ),
        showland=True, landcolor=config.COLORS["bg_page"],
        showocean=True, oceancolor=config.COLORS["bg_page"],
        showlakes=False,
        showcountries=True,
        countrycolor=config.COLORS["primary_light"],
        countrywidth=1.2,
        showcoastlines=True,
        coastlinecolor=config.COLORS["primary_light"],
        coastlinewidth=1.0,
        showsubunits=False,
        showrivers=False,
        bgcolor=config.COLORS["bg_page"],
    )

    # ── 5. Layout ──
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=460,
        font=dict(family="Inter, system-ui, sans-serif", size=11,
                  color=config.COLORS["text"]),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="top", y=-0.02,
            xanchor="center", x=0.5, font_size=9,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.1)", borderwidth=1,
        ),
        paper_bgcolor=config.COLORS["bg_page"],
        plot_bgcolor=config.COLORS["bg_page"],
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

    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    rgb_lo = _hex_to_rgb(config.COLORS["alert"])
    rgb_hi = _hex_to_rgb(config.COLORS["primary_light"])

    def _gradient_color(v):
        """Interpolate from alert (red, low) to primary_light (blue, high)."""
        t = (v - vmin) / span  # 0 = min (red), 1 = max (blue)
        r_lo, g_lo, b_lo = rgb_lo
        r_hi, g_hi, b_hi = rgb_hi
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


# ── Tableaux de données pour le template ──────────────────────

def compute_trend_table(df: pd.DataFrame) -> dict:
    """Calcule les détails OLS+HAC et Sen pour le tableau comparatif."""
    ols = ols_trend_hac(df["gwsa_mm"], df["is_imputed"])
    mk_sen = mann_kendall_sen(df["gwsa_mm"], df["is_imputed"])
    area_m2 = compute_aoi_area_m2(config.AOI_GEOJSON)

    return {
        "ols_slope_mm": f"{ols['slope_mm_yr']:+.2f}",
        "ols_ci": f"[{ols['ci_lower_mm_yr']:+.2f}, {ols['ci_upper_mm_yr']:+.2f}]",
        "ols_pvalue": "< 0.001" if ols["pvalue"] < 0.001 else f"{ols['pvalue']:.3f}",
        "ols_slope_km3": f"{mm_to_km3(ols['slope_mm_yr'], area_m2):+.2f}",
        "sen_slope_mm": f"{mk_sen['sen_slope_mm_yr']:+.2f}",
        "sen_ci": "\u2014",
        "sen_pvalue": "< 0.001" if mk_sen["mk_pvalue"] < 0.001 else f"{mk_sen['mk_pvalue']:.3f}",
        "sen_slope_km3": f"{mm_to_km3(mk_sen['sen_slope_mm_yr'], area_m2):+.2f}",
    }


def compute_forecast_milestones(df: pd.DataFrame) -> list[dict]:
    """Calcule le tableau des jalons de prévision (12, 24, 36, 48, 60 mois)."""
    scenarios = build_scenarios(gwsa_mm=df["gwsa_mm"], validated_mae_mm=config.VALIDATED_MAE_MM)
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
    scenarios = build_scenarios(gwsa_mm=df["gwsa_mm"], validated_mae_mm=config.VALIDATED_MAE_MM)
    forecast_df = scenarios["forecast_df"]

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

    # Gradient alert → primary_light
    def _hex_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    lo_r, lo_g, lo_b = _hex_rgb(config.COLORS["alert"])
    hi_r, hi_g, hi_b = _hex_rgb(config.COLORS["primary_light"])
    colors = []
    for v in vals:
        t = (v - vmin) / span
        r = int(lo_r + (hi_r - lo_r) * t)
        g = int(lo_g + (hi_g - lo_g) * t)
        b = int(lo_b + (hi_b - lo_b) * t)
        colors.append(f"rgb({r},{g},{b})")

    # Compute km³ equivalents for secondary labels
    area_m2 = compute_aoi_area_m2(config.AOI_GEOJSON)
    km3_vals = [v * area_m2 / 1e12 for v in vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(y) for y in combined.index],
        y=combined.values,
        marker_color=colors,
        text=[f"{v:.0f} mm<br><span style='font-size:7px;color:#6B7280'>{k:.1f} km³</span>"
              for v, k in zip(vals, km3_vals)],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="%{x}: %{y:.1f} mm<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=11,
                  color=config.COLORS["text"]),
        margin=dict(l=35, r=10, t=10, b=25),
        height=200,
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
        validated_mae_mm=config.VALIDATED_MAE_MM,
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
                    xanchor="center", x=0.5, font_size=10,
                    entrywidth=130),
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="x unified",
        height=380,
    )
    return fig


# ── Données et figures supplémentaires ──────────────────────────

def compute_scenario_comparison(df: pd.DataFrame) -> list[dict]:
    """Comparaison annuelle des 3 scénarios sur 5 ans (pour le template)."""
    multi = build_multi_scenarios(
        gwsa_mm=df["gwsa_mm"],
        is_imputed=df["is_imputed"],
        validated_mae_mm=config.VALIDATED_MAE_MM,
    )
    area_m2 = compute_aoi_area_m2(config.AOI_GEOJSON)

    rows = []
    for label, sc_df in [
        ("central", multi["central_df"]),
        ("dry", multi["dry_df"]),
        ("wet", multi["wet_df"]),
    ]:
        annual = sc_df.groupby(sc_df["ds"].dt.year)["yhat"].mean()
        for year, val_mm in annual.items():
            val_cm = val_mm / 10.0
            val_km3 = val_mm * area_m2 / 1e12
            rows.append({
                "scenario": label,
                "year": int(year),
                "mm": f"{val_mm:.0f}",
                "cm": f"{val_cm:.1f}",
                "km3": f"{val_km3:.1f}",
            })
    return rows


def make_expert_scenario_figure(df: pd.DataFrame, strings: dict) -> go.Figure:
    """Vue Expert : 3 scénarios détaillés avec bandes IC pour chaque scénario."""
    multi = build_multi_scenarios(
        gwsa_mm=df["gwsa_mm"],
        is_imputed=df["is_imputed"],
        validated_mae_mm=config.VALIDATED_MAE_MM,
    )
    central_df = multi["central_df"]
    dry_df = multi["dry_df"]
    wet_df = multi["wet_df"]
    last_obs_date = multi["last_obs_date"]
    cutoff_date = multi["cutoff_date"]

    obs = df[~df["is_imputed"]].copy()
    obs_start = last_obs_date - pd.DateOffset(years=8)
    obs_recent = obs.loc[obs_start:]

    fig = go.Figure()

    # Full observation history
    fig.add_trace(go.Scatter(
        x=obs_recent.index, y=obs_recent["gwsa_mm"],
        mode="lines",
        name=strings.get("ts_observed", "Observed"),
        line=dict(color=config.COLORS["primary"], width=2),
        hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra></extra>",
    ))

    # CI bands + lines for each scenario
    scenario_specs = [
        (central_df, config.COLORS["primary_light"], "scenario_central", "Trend continuation"),
        (dry_df, config.COLORS["scenario_dry"], "scenario_dry_label", "Dry years"),
        (wet_df, config.COLORS["scenario_wet"], "scenario_wet_label", "Enhanced management"),
    ]
    for sc_df, color, name_key, default_name in scenario_specs:
        # Convert hex color to rgba for fill
        if color.startswith("#"):
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fill_color = f"rgba({r},{g},{b},0.08)"
        else:
            fill_color = config.COLORS["primary_bg"]

        fig.add_trace(go.Scatter(
            x=pd.concat([sc_df["ds"], sc_df["ds"][::-1]]),
            y=pd.concat([sc_df["yhat_upper"], sc_df["yhat_lower"][::-1]]),
            fill="toself", fillcolor=fill_color,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=sc_df["ds"], y=sc_df["yhat"],
            mode="lines",
            name=strings.get(name_key, default_name),
            line=dict(color=color,
                      width=2.5 if "central" in name_key else 2,
                      dash="solid" if "central" in name_key else "dash"),
            hovertemplate="%{x|%b %Y}: %{y:.1f} mm<extra></extra>",
        ))

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
                    xanchor="center", x=0.5, font_size=11,
                    entrywidth=0.22, entrywidthmode="fraction"),
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode="x unified",
        height=420,
    )
    return fig
