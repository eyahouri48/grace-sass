# fichier : tests/test_decomposition.py
"""Tests unitaires pour pipeline/decomposition.py."""

import numpy as np
import pandas as pd
import pytest

from pipeline.decomposition import run_stl, compute_acf_pacf, run_decomposition_diagnostics


# ── Fixture : série synthétique avec tendance + saisonnalité connue 

@pytest.fixture
def synthetic_series():
    """Crée une série de 120 mois (10 ans) :
    tendance linéaire descendante + saisonnalité sinusoïdale + petit bruit.
    """
    dates = pd.date_range("2002-01", periods=120, freq="MS")
    t = np.arange(120)

    trend = -0.5 * t                          # baisse de 0.5 mm/mois
    seasonal = 3.0 * np.sin(2 * np.pi * t / 12)  # amplitude 6 mm (3 × 2)
    noise = np.random.default_rng(42).normal(0, 0.5, 120)

    values = trend + seasonal + noise
    return pd.Series(values, index=dates, name="gwsa_mm")


@pytest.fixture
def flat_series():
    """Série constante (saisonnalité = 0)."""
    dates = pd.date_range("2002-01", periods=60, freq="MS")
    return pd.Series(np.full(60, -10.0), index=dates, name="gwsa_mm")


# ── Tests run_stl ────────────────────────────────────────────────────

def test_stl_reconstruction(synthetic_series):
    """La somme trend + seasonal + resid doit redonner la série originale."""
    stl_df = run_stl(synthetic_series)
    reconstruction = stl_df["trend"] + stl_df["seasonal"] + stl_df["resid"]
    np.testing.assert_allclose(
        stl_df["observed"].values,
        reconstruction.values,
        atol=1e-10,
        err_msg="STL : observed ≠ trend + seasonal + resid"
    )


def test_stl_output_shape(synthetic_series):
    """Le DataFrame de sortie a 4 colonnes et le même nombre de lignes."""
    stl_df = run_stl(synthetic_series)
    assert stl_df.shape == (len(synthetic_series), 4)
    assert list(stl_df.columns) == ["observed", "trend", "seasonal", "resid"]


def test_stl_rejects_nan():
    """STL doit lever une ValueError si la série contient des NaN."""
    dates = pd.date_range("2002-01", periods=24, freq="MS")
    s = pd.Series(np.arange(24, dtype=float), index=dates)
    s.iloc[5] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run_stl(s)


def test_stl_detects_seasonality(synthetic_series):
    """Sur une série avec saisonnalité injectée (amplitude 6 mm),
    STL doit retrouver une amplitude saisonnière significative."""
    stl_df = run_stl(synthetic_series)
    amp = stl_df["seasonal"].max() - stl_df["seasonal"].min()
    # L'amplitude injectée est 6 mm — STL doit retrouver au moins 4 mm
    assert amp > 4.0, f"Amplitude saisonnière trop faible : {amp:.1f} mm"


def test_stl_flat_series(flat_series):
    """Sur une série constante, la saisonnalité et le résidu sont ≈ 0."""
    stl_df = run_stl(flat_series)
    assert stl_df["seasonal"].abs().max() < 1e-10
    assert stl_df["resid"].abs().max() < 1e-10


# ── Tests compute_acf_pacf ───────────────────────────────────────────

def test_acf_pacf_shapes(synthetic_series):
    """Les tableaux ACF/PACF ont la bonne taille."""
    stl_df = run_stl(synthetic_series)
    result = compute_acf_pacf(stl_df["resid"], nlags=24)

    # nlags + 1 valeurs (lag 0 inclus)
    assert len(result["acf_values"]) == 25
    assert len(result["pacf_values"]) == 25
    # Intervalles de confiance : (nlags+1, 2)
    assert result["acf_confint"].shape == (25, 2)
    assert result["pacf_confint"].shape == (25, 2)


def test_acf_lag0_is_one(synthetic_series):
    """L'ACF au lag 0 vaut toujours 1 (corrélation de la série avec elle-même)."""
    stl_df = run_stl(synthetic_series)
    result = compute_acf_pacf(stl_df["resid"])
    assert abs(result["acf_values"][0] - 1.0) < 1e-10


# ── Test du wrapper ──────────────────────────────────────────────────

def test_diagnostics_wrapper(synthetic_series):
    """Le wrapper retourne les clés attendues avec des types corrects."""
    result = run_decomposition_diagnostics(synthetic_series)

    assert "stl_df" in result
    assert "seasonal_amplitude_mm" in result
    assert "acf_pacf" in result

    assert isinstance(result["stl_df"], pd.DataFrame)
    assert isinstance(result["seasonal_amplitude_mm"], float)
    assert result["seasonal_amplitude_mm"] >= 0