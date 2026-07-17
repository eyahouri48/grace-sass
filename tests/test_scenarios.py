# fichier : tests/test_scenarios.py
"""Tests unitaires pour pipeline/scenarios.py."""

import pandas as pd
import numpy as np
import pytest

from pipeline.scenarios import build_scenarios, get_scenario_summary


@pytest.fixture
def fake_gwsa():
    """Série synthétique de 120 mois avec tendance linéaire descendante."""
    dates = pd.date_range("2002-04-01", periods=120, freq="MS")
    # Tendance : -1 mm/mois + bruit léger
    values = -np.arange(120, dtype=float) + np.random.default_rng(42).normal(0, 2, 120)
    s = pd.Series(values, index=dates, name="gwsa_mm")
    s.index.name = "date"
    return s


class TestBuildScenarios:
    """Tests de la fonction build_scenarios."""

    def test_output_structure(self, fake_gwsa):
        """Le dictionnaire retourné contient toutes les clés attendues."""
        result = build_scenarios(fake_gwsa, validated_mae_mm=5.0)

        assert "forecast_df" in result
        assert "last_obs_date" in result
        assert "validated_horizon" in result
        assert "scenario_horizon" in result
        assert "validated_mae_mm" in result
        assert "cutoff_date" in result
        assert "warnings" in result

    def test_zone_labels(self, fake_gwsa):
        """Le DataFrame contient exactement les zones 'validated' et 'extrapolation'."""
        result = build_scenarios(
            fake_gwsa, validated_mae_mm=5.0,
            forecast_horizon=24, scenario_horizon=60,
        )
        zones = set(result["forecast_df"]["zone"].unique())
        assert zones == {"validated", "extrapolation"}

    def test_validated_count(self, fake_gwsa):
        """La zone validée contient exactement forecast_horizon mois."""
        result = build_scenarios(
            fake_gwsa, validated_mae_mm=5.0,
            forecast_horizon=24, scenario_horizon=60,
        )
        n_validated = (result["forecast_df"]["zone"] == "validated").sum()
        assert n_validated == 24

    def test_extrapolation_count(self, fake_gwsa):
        """La zone extrapolation = scenario_horizon − forecast_horizon mois."""
        result = build_scenarios(
            fake_gwsa, validated_mae_mm=5.0,
            forecast_horizon=24, scenario_horizon=60,
        )
        n_extrap = (result["forecast_df"]["zone"] == "extrapolation").sum()
        assert n_extrap == 60 - 24

    def test_ci_widens_over_time(self, fake_gwsa):
        """L'intervalle de confiance s'élargit avec l'horizon."""
        result = build_scenarios(fake_gwsa, validated_mae_mm=5.0)
        df = result["forecast_df"]
        df["ci_width"] = df["yhat_upper"] - df["yhat_lower"]
        # Comparer la largeur moyenne de l'IC : extrapolation > validée
        width_val = df.loc[df["zone"] == "validated", "ci_width"].mean()
        width_ext = df.loc[df["zone"] == "extrapolation", "ci_width"].mean()
        assert width_ext > width_val

    def test_warnings_bilingual(self, fake_gwsa):
        """Les avertissements existent en EN et FR, avec les deux zones."""
        result = build_scenarios(fake_gwsa, validated_mae_mm=5.0)
        for lang in ("en", "fr"):
            assert lang in result["warnings"]
            assert "validated" in result["warnings"][lang]
            assert "extrapolation" in result["warnings"][lang]

    def test_mae_in_warnings(self, fake_gwsa):
        """La MAE validée apparaît dans le texte d'avertissement."""
        result = build_scenarios(fake_gwsa, validated_mae_mm=7.3)
        assert "7.3" in result["warnings"]["en"]["validated"]
        assert "7.3" in result["warnings"]["fr"]["validated"]


class TestGetScenarioSummary:
    """Tests de la fonction get_scenario_summary."""

    def test_milestones_present(self, fake_gwsa):
        """Le tableau contient les jalons 12, 24, 36, 48, 60."""
        scenarios = build_scenarios(
            fake_gwsa, validated_mae_mm=5.0,
            forecast_horizon=24, scenario_horizon=60,
        )
        summary = get_scenario_summary(scenarios)
        assert list(summary["horizon_months"]) == [12, 24, 36, 48, 60]

    def test_zone_assignment_in_summary(self, fake_gwsa):
        """Les jalons 12 et 24 sont 'validated', 36+ sont 'extrapolation'."""
        scenarios = build_scenarios(
            fake_gwsa, validated_mae_mm=5.0,
            forecast_horizon=24, scenario_horizon=60,
        )
        summary = get_scenario_summary(scenarios)
        assert summary.loc[summary["horizon_months"] == 12, "zone"].iloc[0] == "validated"
        assert summary.loc[summary["horizon_months"] == 24, "zone"].iloc[0] == "validated"
        assert summary.loc[summary["horizon_months"] == 36, "zone"].iloc[0] == "extrapolation"

    def test_shorter_scenario_horizon(self, fake_gwsa):
        """Si scenario_horizon=36, les jalons 48 et 60 n'apparaissent pas."""
        scenarios = build_scenarios(
            fake_gwsa, validated_mae_mm=5.0,
            forecast_horizon=24, scenario_horizon=36,
        )
        summary = get_scenario_summary(scenarios)
        assert 48 not in summary["horizon_months"].values
        assert 60 not in summary["horizon_months"].values