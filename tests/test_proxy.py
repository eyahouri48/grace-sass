# fichier : tests/test_proxy.py
"""
Tests unitaires — pipeline/proxy.py (Option B-lite, spec §3).

Fixtures synthétiques, zéro réseau.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.proxy import anomalize_gldas, compute_gwsa, find_baseline_months


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def twsa_cm_with_gaps():
    """Série GRACE synthétique (cm) avec des mois manquants dans la baseline.

    24 mois dans 2004-2009 (= fenêtre baseline), dont 2 sont NaN.
    On vérifie que find_baseline_months ne retient que les 22 mois réels.
    Quelques mois hors baseline pour vérifier qu'ils sont ignorés.
    """
    # Créer un index mensuel de 2003-06 à 2010-06
    index = pd.date_range("2003-06-01", "2010-06-01", freq="MS")
    values = np.arange(len(index), dtype=float) * 0.1  # valeurs arbitraires

    twsa = pd.Series(values, index=index, name="twsa_cm")

    # Supprimer 2 mois dans la baseline (simuler des lacunes GRACE)
    twsa.loc["2005-03-01"] = np.nan
    twsa.loc["2007-11-01"] = np.nan

    # Supprimer 1 mois hors baseline (ne doit PAS affecter le résultat)
    twsa.loc["2003-09-01"] = np.nan

    return twsa


@pytest.fixture
def gldas_mm_simple():
    """Série GLDAS synthétique (mm) avec des valeurs connues.

    Valeur constante de 500 mm → la moyenne baseline sera 500,
    et l'anomalie sera 0 partout (facile à vérifier).
    """
    index = pd.date_range("2003-01-01", "2010-12-01", freq="MS")
    return pd.Series(500.0, index=index, name="gldas_mm")


@pytest.fixture
def gldas_mm_variable():
    """Série GLDAS avec un step : 400 mm avant 2007, 600 mm après.

    Permet de vérifier que l'anomalisation produit des valeurs
    négatives (avant le step) et positives (après).
    """
    index = pd.date_range("2003-01-01", "2010-12-01", freq="MS")
    values = np.where(index < "2007-01-01", 400.0, 600.0)
    return pd.Series(values, index=index, name="gldas_mm")


# ── Tests find_baseline_months ──────────────────────────────────


class TestFindBaselineMonths:
    """Vérifier que seuls les mois RÉELS de GRACE dans 2004-2009 sont retenus."""

    def test_exclut_mois_nan(self, twsa_cm_with_gaps):
        """Les 2 mois NaN dans 2004-2009 ne doivent PAS être dans la baseline."""
        baseline = find_baseline_months(twsa_cm_with_gaps)

        assert pd.Timestamp("2005-03-01") not in baseline
        assert pd.Timestamp("2007-11-01") not in baseline

    def test_exclut_mois_hors_fenetre(self, twsa_cm_with_gaps):
        """Les mois avant 2004 et après 2009 ne sont pas dans la baseline."""
        baseline = find_baseline_months(twsa_cm_with_gaps)

        # Aucun mois hors [2004-01, 2009-12]
        assert all(m >= pd.Timestamp("2004-01-01") for m in baseline)
        assert all(m <= pd.Timestamp("2009-12-01") for m in baseline)

    def test_nombre_mois_reels(self, twsa_cm_with_gaps):
        """72 mois calendaires dans 2004-2009, moins 2 NaN = 70 attendus."""
        baseline = find_baseline_months(twsa_cm_with_gaps)

        # Compter les mois calendaires dans la fenêtre qui ont une valeur
        window = twsa_cm_with_gaps.loc["2004-01":"2009-12"]
        expected = window.dropna().index
        assert len(baseline) == len(expected)


# ── Tests anomalize_gldas ───────────────────────────────────────


class TestAnomalizeGldas:
    """Vérifier la conversion en anomalie par rapport à la baseline GRACE."""

    def test_anomalie_constante_donne_zero(self, gldas_mm_simple, twsa_cm_with_gaps):
        """Si GLDAS est constant (500 mm), l'anomalie est 0 partout."""
        baseline = find_baseline_months(twsa_cm_with_gaps)
        anom = anomalize_gldas(gldas_mm_simple, baseline)

        # Toutes les valeurs doivent être ~0
        assert np.allclose(anom.values, 0.0, atol=1e-10)

    def test_anomalie_variable_signe(self, gldas_mm_variable, twsa_cm_with_gaps):
        """Si GLDAS passe de 400 à 600 mm, l'anomalie change de signe.

        La moyenne baseline (2004-2009) mélange 400 et 600,
        donc les mois à 400 auront une anomalie négative
        et les mois à 600 une anomalie positive.
        """
        baseline = find_baseline_months(twsa_cm_with_gaps)
        anom = anomalize_gldas(gldas_mm_variable, baseline)

        # Avant le step (400 mm) → anomalie < 0
        assert anom.loc["2004-06-01"] < 0
        # Après le step (600 mm) → anomalie > 0
        assert anom.loc["2009-06-01"] > 0

    def test_nom_serie(self, gldas_mm_simple, twsa_cm_with_gaps):
        """La série retournée doit s'appeler 'gldas_anom_mm'."""
        baseline = find_baseline_months(twsa_cm_with_gaps)
        anom = anomalize_gldas(gldas_mm_simple, baseline)
        assert anom.name == "gldas_anom_mm"


# ── Tests compute_gwsa ──────────────────────────────────────────


class TestComputeGwsa:
    """Vérifier la soustraction TWSA (cm→mm) − GLDAS anomalie (mm)."""

    def test_conversion_cm_vers_mm(self):
        """TWSA 1.0 cm avec GLDAS anomalie 0 → GWSA doit valoir 10.0 mm."""
        index = pd.date_range("2005-01-01", periods=3, freq="MS")
        twsa_cm = pd.Series([1.0, 2.0, 3.0], index=index, name="twsa_cm")
        gldas_anom = pd.Series([0.0, 0.0, 0.0], index=index, name="gldas_anom_mm")

        gwsa = compute_gwsa(twsa_cm, gldas_anom)

        # 1 cm × 10 = 10 mm, 2 cm × 10 = 20 mm, etc.
        np.testing.assert_array_almost_equal(gwsa.values, [10.0, 20.0, 30.0])

    def test_soustraction_correcte(self):
        """TWSA 2.0 cm (= 20 mm) − GLDAS anomalie 5.0 mm → GWSA = 15.0 mm."""
        index = pd.date_range("2005-01-01", periods=2, freq="MS")
        twsa_cm = pd.Series([2.0, 3.0], index=index, name="twsa_cm")
        gldas_anom = pd.Series([5.0, 10.0], index=index, name="gldas_anom_mm")

        gwsa = compute_gwsa(twsa_cm, gldas_anom)

        # (2×10 − 5) = 15, (3×10 − 10) = 20
        np.testing.assert_array_almost_equal(gwsa.values, [15.0, 20.0])

    def test_alignement_inner_join(self):
        """Si les index ne couvrent pas les mêmes mois, on ne garde que l'intersection."""
        idx_grace = pd.date_range("2005-01-01", periods=4, freq="MS")
        idx_gldas = pd.date_range("2005-02-01", periods=4, freq="MS")
        # Intersection = 2005-02, 2005-03, 2005-04 (3 mois)

        twsa = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx_grace, name="twsa_cm")
        gldas = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx_gldas, name="gldas_anom_mm")

        gwsa = compute_gwsa(twsa, gldas)

        assert len(gwsa) == 3  # seuls les 3 mois communs
        assert gwsa.index[0] == pd.Timestamp("2005-02-01")
        assert gwsa.index[-1] == pd.Timestamp("2005-04-01")

    def test_nom_serie(self):
        """La série retournée doit s'appeler 'gwsa_mm'."""
        index = pd.date_range("2005-01-01", periods=2, freq="MS")
        twsa = pd.Series([1.0, 1.0], index=index, name="twsa_cm")
        gldas = pd.Series([0.0, 0.0], index=index, name="gldas_anom_mm")

        gwsa = compute_gwsa(twsa, gldas)
        assert gwsa.name == "gwsa_mm"