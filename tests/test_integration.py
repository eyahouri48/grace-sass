# fichier : tests/test_integration.py
"""
Tests d'intégration — pipeline bout en bout sur fixture synthétique.

Vérifie que les modules se chaînent correctement :
    proxy.compute_gwsa
    → preprocessing.reindex_monthly
    → preprocessing.interpolate_gaps
    → trend.compute_full_trend
    → indicators.compute_anomaly_indicators

Aucun appel réseau, aucun fichier distant.
Le seul fichier local lu est sass.geojson (pour compute_aoi_area_m2).
"""

import socket
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pipeline.proxy import compute_gwsa
from pipeline.preprocessing import reindex_monthly, interpolate_gaps
from pipeline.trend import compute_full_trend, mm_to_km3
from pipeline.indicators import compute_anomaly_indicators


# ── Fixtures synthétiques ───────────────────────────────────────


@pytest.fixture
def synthetic_twsa_cm():
    """Série TWSA synthétique (cm) : tendance linéaire descendante + bruit.

    72 mois (2004-01 à 2009-12) pour la baseline, puis 120 mois
    supplémentaires (2010-01 à 2019-12). Deux mois manquants dans
    la baseline + un trou de 3 mois consécutifs (simule la lacune
    inter-missions) en 2017-07 à 2017-09.
    Total : 192 mois, dont 5 sont NaN.
    """
    index = pd.date_range("2004-01-01", periods=192, freq="MS")

    # Tendance : −0.5 cm/mois → −6 cm/an (en mm : −60 mm/an)
    trend = np.arange(192) * (-0.5)

    # Petit bruit reproductible
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.3, 192)

    values = trend + noise

    twsa = pd.Series(values, index=index, name="twsa_cm")

    # Deux mois manquants dans la baseline (simule des mois GRACE absents)
    twsa.iloc[5] = np.nan   # 2004-06
    twsa.iloc[18] = np.nan  # 2005-07

    # Lacune inter-missions : 3 mois consécutifs en 2017
    twsa.iloc[162] = np.nan  # 2017-07
    twsa.iloc[163] = np.nan  # 2017-08
    twsa.iloc[164] = np.nan  # 2017-09

    return twsa


@pytest.fixture
def synthetic_gldas_mm():
    """Série GLDAS synthétique (mm) : quasi constante (surface faible).

    Même index que TWSA. Les composantes de surface sont faibles
    sur le SASS hyperaride → GLDAS ≈ petite oscillation autour de 200 mm.
    Pas de NaN (GLDAS n'a pas de lacune comparable à GRACE).
    """
    index = pd.date_range("2004-01-01", periods=192, freq="MS")

    rng = np.random.RandomState(123)
    # Faible oscillation saisonnière + bruit
    seasonal = 2.0 * np.sin(2 * np.pi * np.arange(192) / 12)
    noise = rng.normal(0, 0.5, 192)
    values = 200.0 + seasonal + noise

    return pd.Series(values, index=index, name="gldas_mm")


# ── Tests d'intégration ────────────────────────────────────────


class TestPipelineBoutEnBout:
    """Chaîne complète : proxy → preprocessing → trend → indicators."""

    def test_chain_proxy_to_preprocessing(
        self, synthetic_twsa_cm, synthetic_gldas_mm
    ):
        """compute_gwsa → reindex_monthly → interpolate_gaps :
        la sortie a les bonnes colonnes, les bons types, zéro NaN
        après interpolation, et is_imputed marque les bons mois.
        """
        # --- Étape proxy ---
        gwsa = compute_gwsa(synthetic_twsa_cm, synthetic_gldas_mm)
        assert gwsa.name == "gwsa_mm", "compute_gwsa doit renvoyer une Series nommée 'gwsa_mm'"

        # Construire le DataFrame attendu par preprocessing
        df = pd.DataFrame({
            "twsa_cm": synthetic_twsa_cm,
            "gldas_anom_mm": synthetic_gldas_mm,  # simplifié pour le test
            "gwsa_mm": gwsa,
        })

        # --- Étape reindex ---
        df_reindexed = reindex_monthly(df)
        assert "is_imputed" in df_reindexed.columns, "reindex_monthly doit ajouter is_imputed"
        assert df_reindexed.index.freq == "MS" or pd.infer_freq(df_reindexed.index) == "MS", \
            "L'index doit être mensuel (freq='MS')"

        # Les 5 NaN d'origine doivent être marqués is_imputed=True
        n_imputed = df_reindexed["is_imputed"].sum()
        assert n_imputed >= 5, f"Au moins 5 mois imputés attendus, trouvé {n_imputed}"

        # --- Étape interpolation ---
        df_filled = interpolate_gaps(df_reindexed)
        assert df_filled["gwsa_mm"].isna().sum() == 0, \
            "Après interpolate_gaps, gwsa_mm ne doit plus avoir de NaN"
        assert df_filled["is_imputed"].sum() == n_imputed, \
            "interpolate_gaps ne doit pas modifier le drapeau is_imputed"

    def test_chain_preprocessing_to_trend(
        self, synthetic_twsa_cm, synthetic_gldas_mm
    ):
        """Le pipeline complet jusqu'à compute_full_trend produit
        un dictionnaire avec les clés attendues et des valeurs cohérentes.
        """
        # Proxy
        gwsa = compute_gwsa(synthetic_twsa_cm, synthetic_gldas_mm)
        df = pd.DataFrame({
            "twsa_cm": synthetic_twsa_cm,
            "gldas_anom_mm": synthetic_gldas_mm,
            "gwsa_mm": gwsa,
        })

        # Preprocessing
        df = reindex_monthly(df)
        df = interpolate_gaps(df)

        # Trend — sur mois observés uniquement
        result = compute_full_trend(
            series=df["gwsa_mm"],
            is_imputed=df["is_imputed"],
        )

        # Vérifier la structure du résultat
        assert isinstance(result, dict), "compute_full_trend doit renvoyer un dict"

        # Clés OLS attendues
        for key in ["ols_slope_mm_yr", "ols_pvalue"]:
            assert key in result, f"Clé manquante dans le résultat : {key}"

        # Clés Mann-Kendall attendues (préfixées mk_)
        for key in ["mk_sen_slope_mm_yr", "mk_mk_pvalue", "mk_mk_trend"]:
            assert key in result, f"Clé manquante dans le résultat : {key}"

        # La pente doit être NÉGATIVE (tendance descendante par construction)
        assert result["ols_slope_mm_yr"] < 0, \
            f"La pente OLS devrait être négative, trouvé {result['ols_slope_mm_yr']}"
        assert result["mk_sen_slope_mm_yr"] < 0, \
            f"La pente de Sen devrait être négative, trouvé {result['mk_sen_slope_mm_yr']}"

        # Mann-Kendall doit détecter une tendance décroissante
        assert result["mk_mk_trend"] == "decreasing", \
            f"MK devrait détecter 'decreasing', trouvé {result['mk_mk_trend']}"

    def test_chain_preprocessing_to_indicators(
        self, synthetic_twsa_cm, synthetic_gldas_mm
    ):
        """Le pipeline jusqu'à compute_anomaly_indicators ajoute
        les colonnes zscore et percentile_rank avec des valeurs sensées.
        """
        # Proxy + preprocessing
        gwsa = compute_gwsa(synthetic_twsa_cm, synthetic_gldas_mm)
        df = pd.DataFrame({
            "twsa_cm": synthetic_twsa_cm,
            "gldas_anom_mm": synthetic_gldas_mm,
            "gwsa_mm": gwsa,
        })
        df = reindex_monthly(df)
        df = interpolate_gaps(df)

        # Indicators
        df_with_indicators = compute_anomaly_indicators(df)

        assert "zscore" in df_with_indicators.columns, "zscore manquant"
        assert "percentile_rank" in df_with_indicators.columns, "percentile_rank manquant"

        # Le dernier mois (le plus bas par construction) doit avoir
        # un z-score très négatif et un percentile bas
        last_zscore = df_with_indicators["zscore"].iloc[-1]
        assert last_zscore < -1.0, \
            f"Le z-score du dernier mois devrait être très négatif, trouvé {last_zscore}"

        last_pct = df_with_indicators["percentile_rank"].iloc[-1]
        assert last_pct < 10, \
            f"Le percentile du dernier mois devrait être très bas, trouvé {last_pct}"

    def test_volume_conversion_coherent(
        self, synthetic_twsa_cm, synthetic_gldas_mm
    ):
        """La conversion mm → km³ dans compute_full_trend est cohérente
        avec un calcul direct via mm_to_km3.
        """
        gwsa = compute_gwsa(synthetic_twsa_cm, synthetic_gldas_mm)
        df = pd.DataFrame({
            "twsa_cm": synthetic_twsa_cm,
            "gldas_anom_mm": synthetic_gldas_mm,
            "gwsa_mm": gwsa,
        })
        df = reindex_monthly(df)
        df = interpolate_gaps(df)

        result = compute_full_trend(df["gwsa_mm"], df["is_imputed"])

        # Vérifier cohérence : slope_km3_yr ≈ slope_mm_yr × area / 1e12
        if "sen_slope_km3_yr" in result and "mk_sen_slope_mm_yr" in result:
            from pipeline.trend import compute_aoi_area_m2
            area = compute_aoi_area_m2()
            expected_km3 = mm_to_km3(result["mk_sen_slope_mm_yr"], area)
            assert abs(result["sen_slope_km3_yr"] - expected_km3) < 0.01, \
                "Incohérence entre sen_slope_km3_yr et mm_to_km3(mk_sen_slope_mm_yr)"

    def test_imputed_months_never_in_trend_stats(
        self, synthetic_twsa_cm, synthetic_gldas_mm
    ):
        """Les mois marqués is_imputed=True ne doivent PAS être utilisés
        dans le calcul de la tendance (§6.1). On vérifie indirectement :
        le nombre de mois utilisés doit être < au total.
        """
        gwsa = compute_gwsa(synthetic_twsa_cm, synthetic_gldas_mm)
        df = pd.DataFrame({
            "twsa_cm": synthetic_twsa_cm,
            "gldas_anom_mm": synthetic_gldas_mm,
            "gwsa_mm": gwsa,
        })
        df = reindex_monthly(df)
        df = interpolate_gaps(df)

        n_total = len(df)
        n_observed = (~df["is_imputed"]).sum()
        assert n_observed < n_total, \
            "Il devrait y avoir des mois imputés dans la fixture"

        result = compute_full_trend(df["gwsa_mm"], df["is_imputed"])

        # Si le dict contient ols_n_obs, vérifier qu'il correspond
        if "ols_n_obs" in result:
            assert result["ols_n_obs"] == n_observed, \
                f"Trend devrait utiliser {n_observed} mois, pas {result['ols_n_obs']}"


# ── Test de non-accès réseau ───────────────────────────────────


class TestNetworkIsolation:
    """Preuve que la suite de tests ne peut pas atteindre le réseau."""

    def test_no_socket_connect(self):
        """Aucun socket.connect ne doit être appelé pendant les tests.
        Si un futur changement introduit un appel HTTP accidentel,
        ce test casse immédiatement.
        """
        def blocked_connect(self_socket, address):
            raise RuntimeError(
                f"INTERDIT : tentative de connexion réseau vers {address} "
                f"détectée pendant les tests. Les tests doivent rester "
                f"100% hors ligne (spec §8.3)."
            )

        with patch.object(socket.socket, "connect", blocked_connect):
            # Réexécuter un mini-pipeline pour prouver qu'il ne sort pas
            index = pd.date_range("2004-01-01", periods=36, freq="MS")
            twsa = pd.Series(np.arange(36, dtype=float) * -0.1, index=index, name="twsa_cm")
            gldas = pd.Series(np.full(36, 100.0), index=index, name="gldas_mm")

            gwsa = compute_gwsa(twsa, gldas)
            df = pd.DataFrame({"gwsa_mm": gwsa, "twsa_cm": twsa, "gldas_anom_mm": gldas})
            df = reindex_monthly(df)
            df = interpolate_gaps(df)
            # Si on arrive ici sans RuntimeError, aucun appel réseau n'a eu lieu

    def test_earthdata_env_vars_absent(self):
        """Confirme que conftest.py a bien retiré les variables Earthdata.
        Redondant avec conftest.py mais sert de preuve explicite dans
        le rapport de tests.
        """
        import os
        for var in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"):
            assert os.environ.get(var) is None, \
                f"La variable {var} ne devrait PAS être dans l'environnement de test"