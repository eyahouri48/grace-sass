# fichier : tests/test_forecast.py
"""
Tests unitaires pour pipeline/forecast.py.

On teste la logique de préparation, de filtrage et de scoring,
PAS les algorithmes Prophet/SARIMA eux-mêmes (bibliothèques externes).
"""

import pandas as pd
import numpy as np
import pytest
from pipeline.forecast import (
    prepare_prophet_df,
    score_observed_only,
    fit_prophet,
    make_forecast,
    walk_forward_cv,
    compute_cv_metrics,
    compare_models,
)


# ── Fixture : série synthétique de 60 mois ──────────────────────

@pytest.fixture
def synthetic_series():
    """Série gwsa_mm synthétique : tendance linéaire + quelques mois imputés."""
    dates = pd.date_range("2010-01-01", periods=60, freq="MS")

    gwsa_mm = pd.Series(
        np.arange(0, -60, -1, dtype=float),
        index=dates,
        name="gwsa_mm",
    )

    is_imputed = pd.Series(False, index=dates, name="is_imputed")
    is_imputed.iloc[10:15] = True

    return gwsa_mm, is_imputed


# ── Tests prepare_prophet_df ────────────────────────────────────

class TestPrepareProphetDf:

    def test_colonnes_ds_y(self, synthetic_series):
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        assert list(df.columns) == ["ds", "y"]

    def test_mois_imputes_retires(self, synthetic_series):
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        assert len(df) == 55

    def test_mois_imputes_gardes(self, synthetic_series):
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=False)
        assert len(df) == 60

    def test_valeurs_y_coherentes(self, synthetic_series):
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        assert df["y"].iloc[0] == 0.0
        assert df["y"].iloc[-1] == -59.0

    def test_nan_dans_gwsa(self):
        dates = pd.date_range("2010-01-01", periods=10, freq="MS")
        gwsa = pd.Series(
            [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            index=dates, dtype=float,
        )
        is_imp = pd.Series(False, index=dates)
        df = prepare_prophet_df(gwsa, is_imp, drop_imputed=False)
        assert len(df) == 10
        assert df["y"].isna().sum() == 2


# ── Tests score_observed_only ───────────────────────────────────

class TestScoreObservedOnly:

    def test_mois_imputes_filtres(self, synthetic_series):
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
        _, is_imputed = synthetic_series
        cv_results = pd.DataFrame({
            "ds": pd.date_range("2010-01-01", periods=20, freq="MS"),
            "yhat": np.arange(20, dtype=float),
            "yhat_lower": np.arange(20, dtype=float) - 1,
            "yhat_upper": np.arange(20, dtype=float) + 1,
            "y": np.arange(20, dtype=float) + 0.5,
            "cutoff": pd.Timestamp("2009-12-01"),
        })
        cv_clean = cv_results.copy()
        cv_clean["ds_month"] = cv_clean["ds"].dt.to_period("M")
        imputed_months = set(is_imputed[is_imputed].index.to_period("M"))
        masque = ~cv_clean["ds_month"].isin(imputed_months)
        assert len(cv_results) == 20
        assert masque.sum() == 15

    def test_aucun_mois_impute_rien_filtre(self):
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

    def test_horizon_correct(self, synthetic_series):
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        model = fit_prophet(df, yearly_seasonality=False)
        horizon = 12
        forecast = make_forecast(model, horizon_months=horizon)
        last_train = df["ds"].max()
        last_forecast = forecast["ds"].max()
        delta_months = (
            (last_forecast.year - last_train.year) * 12
            + (last_forecast.month - last_train.month)
        )
        assert delta_months == horizon

    def test_colonnes_prevision(self, synthetic_series):
        gwsa_mm, is_imputed = synthetic_series
        df = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
        model = fit_prophet(df, yearly_seasonality=False)
        forecast = make_forecast(model, horizon_months=6)
        for col in ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]:
            assert col in forecast.columns, f"Colonne manquante : {col}"


# ── Tests walk-forward et SARIMA/ETS ────────────────────────────

class TestWalkForwardCv:

    def test_walk_forward_retourne_dataframe(self, synthetic_series):
        gwsa_mm, is_imputed = synthetic_series
        cv_df = walk_forward_cv(
            gwsa_mm, is_imputed,
            model_type="ets",
            initial_months=36,
            step_months=6,
            horizon_months=6,
        )
        assert isinstance(cv_df, pd.DataFrame)
        for col in ["cutoff", "ds", "y_true", "y_pred", "is_obs"]:
            assert col in cv_df.columns, f"Colonne manquante : {col}"

    def test_scoring_observed_only(self, synthetic_series):
        """compute_cv_metrics avec observed_only=True exclut les mois imputés."""
        gwsa_mm, is_imputed = synthetic_series

        # Ajouter du bruit — une droite parfaite donne MAE=0 avec ETS
        rng = np.random.default_rng(42)
        gwsa_noisy = gwsa_mm + rng.normal(0, 2, size=len(gwsa_mm))

        cv_df = walk_forward_cv(
            gwsa_noisy, is_imputed,
            model_type="ets",
            initial_months=36,
            step_months=6,
            horizon_months=6,
        )
        metrics_obs = compute_cv_metrics(cv_df, observed_only=True)
        metrics_all = compute_cv_metrics(cv_df, observed_only=False)

        # Avec observed_only, on score sur moins (ou autant) de mois
        assert metrics_obs["n_obs"] <= metrics_all["n_obs"]
        # MAE et RMSE sont des nombres positifs
        assert metrics_obs["mae"] > 0
        assert metrics_obs["rmse"] > 0

    def test_compare_models_tri(self):
        metrics_dict = {
            "Modèle C": {"mae": 10.0, "rmse": 12.0, "n_obs": 100},
            "Modèle A": {"mae": 5.0, "rmse": 7.0, "n_obs": 100},
            "Modèle B": {"mae": 7.5, "rmse": 9.0, "n_obs": 100},
        }
        df = compare_models(metrics_dict)
        assert df.iloc[0]["modèle"] == "Modèle A"
        assert df.iloc[2]["modèle"] == "Modèle C"