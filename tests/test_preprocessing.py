#Tests unitaires — pipeline/proxy.py

"""
Vérifie la réindexation mensuelle, l'interpolation des lacunes,
le drapeau is_imputed, et la détection des blocs de lacunes.
Fixtures synthétiques, zéro réseau.
"""

import pandas as pd
import pytest

from pipeline.preprocessing import (
    find_gap_periods,
    interpolate_gaps,
    reindex_monthly,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def df_with_gaps():
    """DataFrame simulant le cache sass_series.parquet avec des lacunes.

    12 mois (2005-01 à 2005-12), dont 3 mois consécutifs manquants
    (2005-04, 2005-05, 2005-06) → simule une lacune type GRACE.
    Plus 1 mois isolé manquant (2005-10).
    """
    # Index avec des trous (pas tous les mois)
    dates = pd.to_datetime([
        "2005-01-01", "2005-02-01", "2005-03-01",
        # 2005-04, 05, 06 manquants
        "2005-07-01", "2005-08-01", "2005-09-01",
        # 2005-10 manquant
        "2005-11-01", "2005-12-01",
    ])
    gwsa = [10.0, 12.0, 14.0, 22.0, 24.0, 26.0, 30.0, 32.0]
    df = pd.DataFrame({
        "twsa_cm": [v / 10 for v in gwsa],  # valeurs cohérentes
        "gldas_anom_mm": [0.5] * len(gwsa),
        "gwsa_mm": gwsa,
    }, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def df_reindexed(df_with_gaps):
    """DataFrame après réindexation (avec NaN et is_imputed)."""
    return reindex_monthly(df_with_gaps)


# ── Tests reindex_monthly ───────────────────────────────────────


class TestReindexMonthly:
    """Vérifier l'insertion des mois manquants et le drapeau is_imputed."""

    def test_index_complet(self, df_with_gaps):
        """Après réindexation, tous les 12 mois 2005-01 à 2005-12 existent."""
        result = reindex_monthly(df_with_gaps)

        expected_index = pd.date_range("2005-01-01", "2005-12-01", freq="MS", name="date")
        pd.testing.assert_index_equal(result.index, expected_index)

    def test_mois_manquants_sont_nan(self, df_with_gaps):
        """Les 4 mois manquants (04, 05, 06, 10) ont gwsa_mm = NaN."""
        result = reindex_monthly(df_with_gaps)

        for month in ["2005-04-01", "2005-05-01", "2005-06-01", "2005-10-01"]:
            assert pd.isna(result.loc[month, "gwsa_mm"]), f"{month} devrait être NaN"

    def test_mois_existants_inchanges(self, df_with_gaps):
        """Les mois présents dans l'original gardent leurs valeurs."""
        result = reindex_monthly(df_with_gaps)

        assert result.loc["2005-01-01", "gwsa_mm"] == 10.0
        assert result.loc["2005-07-01", "gwsa_mm"] == 22.0
        assert result.loc["2005-12-01", "gwsa_mm"] == 32.0

    def test_is_imputed_flag(self, df_with_gaps):
        """is_imputed=True pour les mois manquants, False pour les existants."""
        result = reindex_monthly(df_with_gaps)

        # Mois existants → False
        assert not result.loc["2005-01-01", "is_imputed"] 
        assert not result.loc["2005-08-01", "is_imputed"] 

        # Mois manquants → True
        assert result.loc["2005-04-01", "is_imputed"] 
        assert result.loc["2005-10-01", "is_imputed"]

    def test_pas_de_perte_de_donnees(self, df_with_gaps):
        """Le nombre de valeurs non-NaN dans gwsa_mm est le même avant/après."""
        result = reindex_monthly(df_with_gaps)

        n_original = df_with_gaps["gwsa_mm"].notna().sum()
        n_after = result["gwsa_mm"].notna().sum()
        assert n_after == n_original


# ── Tests interpolate_gaps ──────────────────────────────────────


class TestInterpolateGaps:
    """Vérifier que l'interpolation comble les trous sans perdre le flag."""

    def test_plus_de_nan_apres_interpolation(self, df_reindexed):
        """Après interpolation, gwsa_mm ne contient plus de NaN."""
        result = interpolate_gaps(df_reindexed)

        assert result["gwsa_mm"].isna().sum() == 0

    def test_is_imputed_inchange(self, df_reindexed):
        """L'interpolation ne modifie PAS le drapeau is_imputed.

        Les mois imputés restent marqués True — c'est critique pour
        la validation croisée (on ne score jamais sur un mois imputé).
        """
        result = interpolate_gaps(df_reindexed)

        # Les mois manquants restent marqués imputés
        assert result.loc["2005-04-01", "is_imputed"] 
        assert result.loc["2005-05-01", "is_imputed"] 

        # Les mois observés restent marqués non-imputés
        assert not result.loc["2005-01-01", "is_imputed"] 

    def test_interpolation_lineaire_coherente(self, df_reindexed):
        """Les valeurs interpolées sont entre les bornes.

        Entre 2005-03 (gwsa=14) et 2005-07 (gwsa=22),
        les mois interpolés doivent être entre 14 et 22.
        """
        result = interpolate_gaps(df_reindexed)

        for month in ["2005-04-01", "2005-05-01", "2005-06-01"]:
            val = result.loc[month, "gwsa_mm"]
            assert 14.0 <= val <= 22.0, (
                f"{month}: gwsa_mm={val}, attendu entre 14 et 22"
            )


# ── Tests find_gap_periods ──────────────────────────────────────


class TestFindGapPeriods:
    """Vérifier la détection des blocs de mois imputés consécutifs."""

    def test_detecte_lacune_3_mois(self, df_reindexed):
        """Le bloc de 3 mois consécutifs (avr-mai-jun) doit être détecté.

        GAP_MIN_CONSECUTIVE = 3 dans config.py, donc ce bloc passe le seuil.
        """
        gaps = find_gap_periods(df_reindexed)

        # Au moins un gap détecté
        assert len(gaps) >= 1

        # Le plus long gap est de 3 mois
        longest = gaps[0]
        assert longest[2] == 3  # nb_mois
        assert longest[0] == pd.Timestamp("2005-04-01")
        assert longest[1] == pd.Timestamp("2005-06-01")

    def test_ignore_lacune_isolee(self, df_reindexed):
        """Le mois isolé (2005-10) fait 1 mois < GAP_MIN_CONSECUTIVE → ignoré."""
        gaps = find_gap_periods(df_reindexed)

        # Aucun gap ne commence en 2005-10
        gap_starts = [g[0] for g in gaps]
        assert pd.Timestamp("2005-10-01") not in gap_starts

    def test_tri_par_duree_decroissante(self, df_reindexed):
        """Les gaps sont triés du plus long au plus court."""
        gaps = find_gap_periods(df_reindexed)

        if len(gaps) > 1:
            durations = [g[2] for g in gaps]
            assert durations == sorted(durations, reverse=True)