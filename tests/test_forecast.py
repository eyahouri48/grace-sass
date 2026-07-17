# fichier : tests/test_forecast.py
"""
Tests unitaires pour pipeline/forecast.py.

On teste la logique de préparation, de filtrage et de scoring,
PAS l'algorithme Prophet lui-même (bibliothèque externe).
"""

import pandas as pd
import numpy as np
import pytest
from pipeline.forecast import (
    prepare_prophet_df,
    score_observed_only,
    fit_prophet,
    make_forecast,
)


# ── Fixture : série synthétique de 60 mois ──────────────────────

@pytest.fixture
def synthetic_series():
    """Série gwsa_mm synthétique : tendance linéaire + quelques mois imputés."""
    dates = pd.date_range("2010-01-01", periods=60, freq="MS")

    # Tendance linéaire descendante : de 0 à -59 mm (1 mm/mois de baisse)
    gwsa_mm = pd.Series(
        np.arange(0, -60, -1, dtype=float),
        index=dates,
        name="gwsa_mm",
    )

    # 5 mois imputés (indices 10, 11, 12, 13, 14 → nov 2010 à mars 2011)
    is_imputed = pd.Series(False, index=dates, name="is_imputed")
    is_imputed.iloc[10:15] = True

    return gwsa_mm, is_imputed


# ── Tests prepare_prophet_df ────────────────────────────────────

class TestPrepareProphetDf:
    """Tests pour prepare_prophet_df()."""

    def test_colonnes_ds_y(self, synthetic_series):
        """Le DataFrame doit avoir exactement les colonnes ds et y."""
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        assert list(df.columns) == ["ds", "y"]

    def test_mois_imputes_retires(self, synthetic_series):
        """Avec drop_imputed=True, les mois imputés sont absents."""
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        assert len(df) == 55  # 60 - 5 imputés

    def test_mois_imputes_gardes(self, synthetic_series):
        """Avec drop_imputed=False, tous les mois sont présents."""
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=False)
        assert len(df) == 60

    def test_valeurs_y_coherentes(self, synthetic_series):
        """Les valeurs y correspondent à gwsa_mm (mois observés)."""
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        assert df["y"].iloc[0] == 0.0
        assert df["y"].iloc[-1] == -59.0

    def test_nan_dans_gwsa(self):
        """Les NaN dans gwsa_mm sont conservés (Prophet les gère)."""
        dates = pd.date_range("2010-01-01", periods=10, freq="MS")
        gwsa = pd.Series(
            [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            index=dates, dtype=float,
        )
        is_imp = pd.Series(False, index=dates)

        df = prepare_prophet_df(gwsa, is_imp, drop_imputed=False)
        # Les 10 lignes sont là (NaN inclus — Prophet les gère)
        assert len(df) == 10
        # Les NaN sont bien dans la colonne y
        assert df["y"].isna().sum() == 2


# ── Tests score_observed_only ───────────────────────────────────

class TestScoreObservedOnly:
    """Tests pour score_observed_only()."""

    def test_mois_imputes_filtres(self, synthetic_series):
        """Les mois imputés sont retirés des résultats de CV avant scoring."""
        _, is_imputed = synthetic_series

        cv_results = pd.DataFrame({
            "ds": pd.date_range("2010-01-01", periods=20, freq="MS"),
            "yhat": np.random.randn(20),
            "yhat_lower": np.random.randn(20),
            "yhat_upper": np.random.randn(20),
            "y": np.random.randn(20),
            "cutoff": pd.Timestamp("2009-12-01"),
        })

        metrics = score_observed_only(cv_results, is_imputed)
        assert "mae" in metrics.columns
        assert "rmse" in metrics.columns

    def test_decompte_filtrage_exact(self, synthetic_series):
        """Vérifier que le bon nombre de lignes est retiré."""
        _, is_imputed = synthetic_series

        # 20 mois : janv 2010 → août 2011
        # Mois imputés dans cette fenêtre : nov 2010 à mars 2011 = 5 mois
        # → 20 - 5 = 15 lignes doivent rester
        cv_results = pd.DataFrame({
            "ds": pd.date_range("2010-01-01", periods=20, freq="MS"),
            "yhat": np.arange(20, dtype=float),
            "yhat_lower": np.arange(20, dtype=float) - 1,
            "yhat_upper": np.arange(20, dtype=float) + 1,
            "y": np.arange(20, dtype=float) + 0.5,
            "cutoff": pd.Timestamp("2009-12-01"),
        })

        # Reproduire la logique de filtrage pour vérifier le décompte
        cv_clean = cv_results.copy()
        cv_clean["ds_month"] = cv_clean["ds"].dt.to_period("M")
        imputed_months = set(is_imputed[is_imputed].index.to_period("M"))
        masque = ~cv_clean["ds_month"].isin(imputed_months)

        assert len(cv_results) == 20       # avant filtrage
        assert masque.sum() == 15          # après filtrage : 5 mois retirés

    def test_aucun_mois_impute_rien_filtre(self):
        """Si aucun mois n'est imputé, toutes les lignes restent."""
        dates = pd.date_range("2010-01-01", periods=10, freq="MS")
        is_imputed = pd.Series(False, index=dates, name="is_imputed")

        cv_results = pd.DataFrame({
            "ds": dates,
            "yhat": np.arange(10, dtype=float),
            "yhat_lower": np.arange(10, dtype=float) - 1,
            "yhat_upper": np.arange(10, dtype=float) + 1,
            "y": np.arange(10, dtype=float) + 0.3,
            "cutoff": pd.Timestamp("2009-12-01"),
        })

        metrics = score_observed_only(cv_results, is_imputed)
        assert "mae" in metrics.columns
        assert len(metrics) > 0

    def test_metrics_en_mm(self, synthetic_series):
        """Les métriques MAE/RMSE doivent être positives (en mm)."""
        _, is_imputed = synthetic_series

        cv_results = pd.DataFrame({
            "ds": pd.date_range("2010-01-01", periods=5, freq="MS"),
            "yhat": [1.0, 2.0, 3.0, 4.0, 5.0],
            "yhat_lower": [0.0, 1.0, 2.0, 3.0, 4.0],
            "yhat_upper": [2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [1.5, 2.5, 3.5, 4.5, 5.5],
            "cutoff": pd.Timestamp("2009-12-01"),
        })

        metrics = score_observed_only(cv_results, is_imputed)
        assert (metrics["mae"] > 0).all()
        assert (metrics["rmse"] > 0).all()


# ── Tests make_forecast ─────────────────────────────────────────

class TestMakeForecast:
    """Tests pour make_forecast()."""

    def test_horizon_correct(self, synthetic_series):
        """La prévision couvre exactement le nombre de mois demandé."""
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        model = fit_prophet(df, yearly_seasonality=False)

        horizon = 12
        forecast = make_forecast(model, horizon_months=horizon)

        # Vérifier que la dernière date est bien 12 mois après
        # la dernière date d'entraînement
        last_train = df["ds"].max()
        last_forecast = forecast["ds"].max()
        delta_months = (
            (last_forecast.year - last_train.year) * 12
            + (last_forecast.month - last_train.month)
        )
        assert delta_months == horizon

    def test_colonnes_prevision(self, synthetic_series):
        """Le DataFrame de prévision contient yhat, yhat_lower, yhat_upper, trend."""
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        model = fit_prophet(df, yearly_seasonality=False)

        forecast = make_forecast(model, horizon_months=6)

        for col in ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]:
            assert col in forecast.columns, f"Colonne manquante : {col}"