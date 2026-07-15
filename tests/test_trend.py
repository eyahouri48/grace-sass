# fichier : tests/test_trend.py
"""Tests unitaires pour le module trend — Tâche 6."""

import numpy as np
import pandas as pd
import pytest

from pipeline.trend import (
    compute_aoi_area_m2,
    mm_to_km3,
    ols_trend_hac,
    mann_kendall_sen,
)


# ────────────────────────────────────────────────────────────
# Test 1 — Conversion mm → km³ avec une surface connue
# ────────────────────────────────────────────────────────────
def test_mm_to_km3_known_area():
    """
    Sur une surface de 1 000 000 km² (= 1e12 m²),
    1 mm d'eau = exactement 1 km³.
    C'est la coïncidence pratique du SASS.
    """
    area_m2 = 1e12  # 1 million de km²
    result = mm_to_km3(1.0, area_m2)
    assert result == pytest.approx(1.0, abs=1e-10)

    # -3.83 mm sur cette surface → -3.83 km³
    result2 = mm_to_km3(-3.83, area_m2)
    assert result2 == pytest.approx(-3.83, abs=1e-10)


# ────────────────────────────────────────────────────────────
# Test 2 — Pente de Sen sur une série linéaire parfaite
# ────────────────────────────────────────────────────────────
def test_sen_slope_linear_series():
    """
    Une série strictement linéaire : y = 100 - 2*mois.
    Toutes les pentes entre paires sont identiques → Sen = -2 par mois.
    Sur 4 ans (48 mois), seasonal_test(period=12) doit
    retourner une pente de -24 mm/an (2 mm/mois × 12 mois/an).

    Note : seasonal_test compare les mêmes mois entre années,
    donc la pente est directement en unité/an.
    """
    # 4 années complètes = 48 mois
    n_months = 48
    dates = pd.date_range("2010-01", periods=n_months, freq="MS")
    # Pente de -2 mm par mois = -24 mm par an
    values = 100 - 2 * np.arange(n_months)
    series = pd.Series(values, index=dates, dtype=float)
    is_imputed = pd.Series(False, index=dates)

    result = mann_kendall_sen(series, is_imputed)

    # La pente de Sen doit être -24 mm/an
    assert result["sen_slope_mm_yr"] == pytest.approx(-24.0, abs=0.1)
    # La tendance doit être 'decreasing'
    assert result["mk_trend"] == "decreasing"
    # Significatif
    assert result["mk_h"] 


# ────────────────────────────────────────────────────────────
# Test 3 — OLS exclut bien les mois imputés
# ────────────────────────────────────────────────────────────
def test_ols_excludes_imputed_months():
    """
    On crée une série de 60 mois avec une tendance linéaire,
    on marque 10 mois comme imputés.
    Le fit OLS ne doit utiliser que les 50 mois observés.
    """
    n = 60
    dates = pd.date_range("2015-01", periods=n, freq="MS")
    values = 100 - 0.5 * np.arange(n)  # pente = -0.5 mm/mois
    series = pd.Series(values, index=dates, dtype=float)

    # 10 mois marqués imputés (du mois 25 au mois 34)
    is_imputed = pd.Series(False, index=dates)
    is_imputed.iloc[25:35] = True

    result = ols_trend_hac(series, is_imputed, maxlags=6)

    # Le nombre d'observations doit être 50, pas 60
    assert result["n_obs"] == 50

    # La pente doit être ~-6 mm/an (-0.5 mm/mois × 12)
    assert result["slope_mm_yr"] == pytest.approx(-6.0, abs=0.5)

    # L'erreur HAC doit être >= l'erreur naïve
    assert result["slope_hac_se"] >= result["slope_naive_se"]


# ────────────────────────────────────────────────────────────
# Test 4 — L'aire géodésique du SASS est ~1 M km²
# ────────────────────────────────────────────────────────────
def test_aoi_area_plausible():
    """
    L'aire du polygone SASS (sass.geojson) doit être
    entre 0.8 et 1.3 million de km².
    """
    area_m2 = compute_aoi_area_m2()
    area_km2 = area_m2 / 1e6

    assert 800_000 < area_km2 < 1_300_000, (
        f"Aire AOI = {area_km2:,.0f} km² — hors plage attendue"
    )


# ────────────────────────────────────────────────────────────
# Test 5 — MK saisonnier ne score pas sur les mois imputés
# ────────────────────────────────────────────────────────────
def test_mk_excludes_imputed():
    """
    On vérifie que MK utilise moins de points quand on marque
    des mois comme imputés. On le vérifie indirectement :
    avec et sans imputés, le z-stat doit différer.
    """
    n = 72  # 6 ans
    dates = pd.date_range("2010-01", periods=n, freq="MS")
    values = 100 - 1.5 * np.arange(n) + np.random.default_rng(42).normal(0, 3, n)
    series = pd.Series(values, index=dates, dtype=float)

    # Sans imputés
    no_imputed = pd.Series(False, index=dates)
    r1 = mann_kendall_sen(series, no_imputed)

    # Avec 12 mois imputés
    with_imputed = pd.Series(False, index=dates)
    with_imputed.iloc[30:42] = True
    r2 = mann_kendall_sen(series, with_imputed)

    # Les z-stats doivent être différents (moins de données)
    assert r1["mk_z"] != r2["mk_z"]
    # Les deux doivent quand même détecter une tendance décroissante
    assert r1["mk_trend"] == "decreasing"
    assert r2["mk_trend"] == "decreasing"