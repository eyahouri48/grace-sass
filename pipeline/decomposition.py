# fichier : pipeline/decomposition.py
"""Décomposition saisonnière-tendance (STL) et diagnostics ACF/PACF.

Ce module sépare la série gwsa_mm en trois composantes :
- tendance (mouvement lent de fond)
- saisonnalité (cycle annuel)
- résidu (fluctuations inexpliquées)

Le résidu sert ensuite à guider le choix des ordres SARIMA
via l'examen de l'ACF et de la PACF.
"""

import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

from pipeline.config import STL_PERIOD, STL_SEASONAL, STL_ROBUST, ACF_NLAGS


def run_stl(series: pd.Series) -> pd.DataFrame:
    """Décompose une série mensuelle via STL.

    Parameters
    ----------
    series : pd.Series
        Série mensuelle CONTINUE (sans NaN) — typiquement gwsa_mm
        après interpolation des lacunes. Index = DatetimeIndex mensuel.

    Returns
    -------
    pd.DataFrame
        Colonnes : 'observed', 'trend', 'seasonal', 'resid'.
        Même index que la série d'entrée.

    Raises
    ------
    ValueError
        Si la série contient des NaN (STL ne les tolère pas).
    """
    #  STL plante sur les NaN ---
    if series.isna().any():
        raise ValueError(
            f"La série contient {series.isna().sum()} NaN. "
            "STL exige une série continue — utiliser la version interpolée."
        )

    # --- Décomposition ---
    stl = STL(
        series,
        period=STL_PERIOD,       # 12 mois = cycle annuel
        seasonal=STL_SEASONAL,   # fenêtre Loess saisonnière (impair, ≥ 7)
        robust=STL_ROBUST,       # résistant aux outliers
    )
    result = stl.fit()

    # --- Assemblage du DataFrame résultat ---
    df = pd.DataFrame({
        "observed": result.observed,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": result.resid,
    }, index=series.index)

    return df


def compute_acf_pacf(
    residuals: pd.Series,
    nlags: int = ACF_NLAGS,
) -> dict:
    """Calcule l'ACF et la PACF d'une série de résidus.

    Parameters
    ----------
    residuals : pd.Series
        Résidus STL (ou toute série stationnaire). Sans NaN.
    nlags : int
        Nombre de lags à calculer (défaut : 36 = 3 ans).

    Returns
    -------
    dict
        Clés :
        - 'acf_values' : np.ndarray des autocorrélations (lag 0 inclus)
        - 'acf_confint' : np.ndarray (nlags+1, 2) bornes de l'IC 95 %
        - 'pacf_values' : np.ndarray des autocorrélations partielles
        - 'pacf_confint' : np.ndarray (nlags+1, 2) bornes de l'IC 95 %
        - 'nlags' : int
    """
    if residuals.isna().any():
        raise ValueError("Les résidus contiennent des NaN.")

    # --- ACF avec intervalle de confiance à 95 % ---
    acf_vals, acf_ci = acf(
        residuals,
        nlags=nlags,
        alpha=0.05,      # IC à 95 %
        fft=True,         # plus rapide
    )

    # --- PACF avec intervalle de confiance à 95 % ---
    pacf_vals, pacf_ci = pacf(
        residuals,
        nlags=nlags,
        alpha=0.05,
        method="ywm",    # Yule-Walker modifié — stable
    )

    return {
        "acf_values": acf_vals,
        "acf_confint": acf_ci,
        "pacf_values": pacf_vals,
        "pacf_confint": pacf_ci,
        "nlags": nlags,
    }


def run_decomposition_diagnostics(series: pd.Series) -> dict:
    """Wrapper complet : STL + ACF/PACF sur le résidu.

    Parameters
    ----------
    series : pd.Series
        Série gwsa_mm interpolée (continue, sans NaN).

    Returns
    -------
    dict
        Clés :
        - 'stl_df' : pd.DataFrame (observed, trend, seasonal, resid)
        - 'seasonal_amplitude_mm' : float (max - min de la composante saisonnière)
        - 'acf_pacf' : dict retourné par compute_acf_pacf()
    """
    # --- Décomposition STL ---
    stl_df = run_stl(series)

    # --- Amplitude saisonnière (un chiffre clé pour le rapport) ---
    seasonal_amplitude = stl_df["seasonal"].max() - stl_df["seasonal"].min()

    # --- ACF/PACF sur le résidu ---
    acf_pacf = compute_acf_pacf(stl_df["resid"])

    return {
        "stl_df": stl_df,
        "seasonal_amplitude_mm": float(seasonal_amplitude),
        "acf_pacf": acf_pacf,
    }