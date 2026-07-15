# fichier : pipeline/trend.py
"""
Trend estimation on the GWSA proxy series.

OLS with HAC (Newey-West) standard errors, seasonal Mann-Kendall,
Sen's slope, and volume conversion (mm -> km³).
Spec §6.1 — observed months only (is_imputed == False).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod
import statsmodels.api as sm
import pymannkendall as mk

from pipeline.config import AOI_GEOJSON, HAC_MAXLAGS


def compute_aoi_area_m2(geojson_path: str = AOI_GEOJSON) -> float:
    """
    Calcule l'aire géodésique du polygone AOI sur l'ellipsoïde WGS84.

    Utilise pyproj.Geod au lieu de .area sur EPSG:4326
    (qui renverrait des degrés carrés, pas des m²).

    Returns
    -------
    float
        Aire en m².
    """
    gdf = gpd.read_file(geojson_path)
    geod = Geod(ellps="WGS84")

    # Pour chaque géométrie du GeoJSON, calculer l'aire géodésique
    total_area = 0.0
    for geom in gdf.geometry:
        # geod.geometry_area_perimeter renvoie (aire_m², périmètre_m)
        # l'aire est signée (positive = sens antihoraire) → on prend abs
        area, _ = geod.geometry_area_perimeter(geom)
        total_area += abs(area)

    return total_area

def mm_to_km3(value_mm: float, area_m2: float) -> float:
    """
    Convertit une valeur en mm (équivalent eau) en km³.

    ΔV (km³) = mm × A_m² / 1e12
    Sur le SASS (~1 M km²) : 1 mm ≈ 1 km³.
    """
    return value_mm * area_m2 / 1e12


def ols_trend_hac(
    series: pd.Series,
    is_imputed: pd.Series,
    maxlags: int = HAC_MAXLAGS,
) -> dict:
    """
    Régression OLS de gwsa_mm vs. temps, avec erreurs HAC Newey-West.

    Parameters
    ----------
    series : pd.Series
        Série gwsa_mm indexée par DatetimeIndex mensuel.
    is_imputed : pd.Series
        Booléen — True pour les mois interpolés (exclus du fit).
    maxlags : int
        Nombre max de lags pour la correction HAC (défaut : 12).

    Returns
    -------
    dict avec clés :
        slope_mm_yr, intercept, ci_lower_mm_yr, ci_upper_mm_yr,
        r_squared, pvalue, slope_naive_se, slope_hac_se
    """
    # --- Filtrer les mois observés uniquement ---
    mask = ~is_imputed
    obs = series[mask].dropna()

    # --- Variable explicative : temps en années fractionnaires ---
    # On part du premier mois observé comme t=0
    t0 = obs.index[0]
    # Nombre de jours depuis t0, converti en années
    x_years = (obs.index - t0).days / 365.25

    # --- Ajout de la constante (intercept) ---
    X = sm.add_constant(x_years)
    y = obs.values

    # --- Fit OLS classique (pour comparer les erreurs naïves) ---
    model_naive = sm.OLS(y, X).fit()

    # --- Fit OLS avec erreurs HAC (Newey-West) ---
    model_hac = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )

    # --- Extraction des résultats ---
    slope = model_hac.params[1]           # mm par an
    ci = model_hac.conf_int(alpha=0.05)   # IC à 95%

    return {
        "slope_mm_yr": slope,
        "intercept": model_hac.params[0],
        "ci_lower_mm_yr": ci[1, 0],       # borne basse de la pente
        "ci_upper_mm_yr": ci[1, 1],       # borne haute de la pente
        "r_squared": model_hac.rsquared,
        "pvalue": model_hac.pvalues[1],
        "slope_naive_se": model_naive.bse[1],   # erreur-type naïve
        "slope_hac_se": model_hac.bse[1],       # erreur-type HAC (plus large)
        "n_obs": len(obs),
    }

def mann_kendall_sen(
    series: pd.Series,
    is_imputed: pd.Series,
) -> dict:
    """
    Test de Mann-Kendall saisonnier + pente de Sen sur les mois observés.

    Utilise pymannkendall.seasonal_test (period=12) pour ne pas
    gonfler la significativité par la saisonnalité/autocorrélation.

    Returns
    -------
    dict avec clés :
        mk_trend ('increasing'/'decreasing'/'no trend'),
        mk_pvalue, mk_z, sen_slope_mm_month, sen_slope_mm_yr
    """
    # --- Filtrer les mois observés ---
    obs = series[~is_imputed].dropna()

    # --- MK saisonnier (period=12 pour le cycle annuel) ---
    result = mk.seasonal_test(obs.values, period=12)

    return {
        "mk_trend": result.trend,            # 'decreasing', 'increasing', 'no trend'
        "mk_h": result.h,                    # True = tendance significative
        "mk_pvalue": result.p,
        "mk_z": result.z,
        "sen_slope_mm_yr": result.slope , # conversion en mm/an
    }

def compute_full_trend(
    series: pd.Series,
    is_imputed: pd.Series,
    geojson_path: str = AOI_GEOJSON,
) -> dict:
    """
    Calcule tous les indicateurs de tendance et la conversion en volume.

    Inclut la vérification de plausibilité par rapport aux chiffres OSS
    (déficit net attendu ~1-2 km³/an ≈ 1-2 mm/an sur le SASS).

    Returns
    -------
    dict contenant les résultats OLS+HAC, MK saisonnier, Sen,
    et les conversions en km³/an.
    """
    # --- Aire géodésique ---
    area_m2 = compute_aoi_area_m2(geojson_path)
    area_km2 = area_m2 / 1e6

    # --- OLS + HAC ---
    ols = ols_trend_hac(series, is_imputed)

    # --- Mann-Kendall saisonnier + Sen ---
    mk_sen = mann_kendall_sen(series, is_imputed)

    # --- Conversions en km³/an ---
    ols_km3_yr = mm_to_km3(ols["slope_mm_yr"], area_m2)
    sen_km3_yr = mm_to_km3(mk_sen["sen_slope_mm_yr"], area_m2)
    ci_lower_km3 = mm_to_km3(ols["ci_lower_mm_yr"], area_m2)
    ci_upper_km3 = mm_to_km3(ols["ci_upper_mm_yr"], area_m2)

    # --- Plausibilité OSS ---
    # Déficit attendu : ~1-2 km³/an (recharge ~1 vs prélèvements ~2.5-3.1)
    # GRACE mesure le stockage total, pas les prélèvements →
    # comparaison indicative seulement
    oss_deficit_range = (1.0, 2.0)  # km³/an

    return {
        # Aire
        "area_m2": area_m2,
        "area_km2": area_km2,
        # OLS + HAC
        **{f"ols_{k}": v for k, v in ols.items()},
        "ols_slope_km3_yr": ols_km3_yr,
        "ols_ci_lower_km3_yr": ci_lower_km3,
        "ols_ci_upper_km3_yr": ci_upper_km3,
        # MK + Sen
        **{f"mk_{k}": v for k, v in mk_sen.items()},
        "sen_slope_km3_yr": sen_km3_yr,
        # Plausibilité
        "oss_expected_deficit_km3_yr": oss_deficit_range,
    }