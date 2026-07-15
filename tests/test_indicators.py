# fichier : tests/test_indicators.py
"""Tests unitaires pour le module indicators (z-score, percentile)."""

import numpy as np
import pandas as pd
import pytest

from pipeline.indicators import (
    compute_anomaly_indicators,
    compute_percentile_rank,
    compute_zscore,
)


@pytest.fixture
def sample_df():
    """DataFrame de test avec 24 mois, dont la référence 2004-2009 contient
    les 12 premiers mois (tous observés), et les 12 suivants sont la suite."""
    dates = pd.date_range("2004-01", periods=24, freq="MS")
    # Valeurs linéaires décroissantes : 10, 9, 8, ..., -13
    values = np.arange(10, -14, -1, dtype=float)
    is_imputed = np.array([False] * 24)
    # On rend le mois 15 imputé pour tester l'exclusion
    is_imputed[14] = True
    return pd.DataFrame(
        {"gwsa_mm": values, "is_imputed": is_imputed},
        index=dates,
    )


class TestZscore:
    """Tests du z-score."""

    def test_baseline_mean_zscore_near_zero(self, sample_df):
        """Le z-score moyen des mois de référence observés doit être ~0."""
        zs = compute_zscore(sample_df["gwsa_mm"], sample_df["is_imputed"])
        # Mois de référence = 2004-01 à 2005-12 (les 24 premiers dans la
        # fixture, mais baseline = 2004-01 à 2009-12, donc les 12 premiers
        # car la série s'arrête avant 2009)
        # Ici la baseline va de 2004-01 à 2005-12 (tout ce qui est <= 2009-12)
        # → les 24 mois sont dans la baseline puisque 2005-12 < 2009-12
        # Correction : les 24 mois vont de 2004-01 à 2005-12 — tous dans la baseline
        baseline_zs = zs.loc["2004-01":"2009-12"]
        # On ne garde que les non-imputés
        mask = ~sample_df["is_imputed"].loc["2004-01":"2009-12"]
        mean_zs = baseline_zs[mask].mean()
        assert abs(mean_zs) < 1e-10, f"Moyenne z-score baseline = {mean_zs}, attendu ~0"

    def test_zscore_sign_decreasing_series(self, sample_df):
        """Sur une série décroissante, les derniers mois ont un z-score négatif."""
        zs = compute_zscore(sample_df["gwsa_mm"], sample_df["is_imputed"])
        # Le dernier mois (valeur = -13) doit avoir un z-score très négatif
        assert zs.iloc[-1] < 0

    def test_zscore_constant_series_returns_nan(self):
        """Si la série est constante, sigma=0 → z-score = NaN."""
        dates = pd.date_range("2004-01", periods=12, freq="MS")
        gwsa = pd.Series(5.0, index=dates, name="gwsa_mm")
        imputed = pd.Series(False, index=dates)
        zs = compute_zscore(gwsa, imputed)
        assert zs.isna().all()


class TestPercentileRank:
    """Tests du rang percentile."""

    def test_max_value_near_100(self, sample_df):
        """La valeur maximale observée doit avoir un percentile ≈ 100."""
        pct = compute_percentile_rank(sample_df["gwsa_mm"], sample_df["is_imputed"])
        # La valeur max (10) est le premier mois, qui est observé
        max_idx = sample_df["gwsa_mm"].idxmax()
        assert pct.loc[max_idx] > 95, f"Percentile du max = {pct.loc[max_idx]}"

    def test_min_value_near_0(self, sample_df):
        """La valeur minimale observée doit avoir un percentile bas."""
        pct = compute_percentile_rank(sample_df["gwsa_mm"], sample_df["is_imputed"])
        min_idx = sample_df["gwsa_mm"].idxmin()
        # Le min est observé (le dernier mois, index 23, non imputé)
        assert pct.loc[min_idx] < 10, f"Percentile du min = {pct.loc[min_idx]}"

    def test_imputed_month_still_gets_percentile(self, sample_df):
        """Un mois imputé reçoit quand même un percentile (calculé contre les observés)."""
        pct = compute_percentile_rank(sample_df["gwsa_mm"], sample_df["is_imputed"])
        imputed_idx = sample_df.index[14]
        assert not np.isnan(pct.loc[imputed_idx])


class TestAnomalyIndicators:
    """Test de la fonction enveloppe."""

    def test_columns_added(self, sample_df):
        """compute_anomaly_indicators ajoute zscore et percentile_rank."""
        result = compute_anomaly_indicators(sample_df)
        assert "zscore" in result.columns
        assert "percentile_rank" in result.columns
        # Les colonnes d'origine sont préservées
        assert "gwsa_mm" in result.columns
        assert "is_imputed" in result.columns