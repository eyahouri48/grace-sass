# fichier : pipeline/indicators.py
"""Indicateurs d'anomalie pour non-spécialistes (z-score, rang percentile).

Calcule le z-score et le rang percentile de la série gwsa_mm
par rapport à la période de référence 2004-2009, en utilisant
uniquement les mois observés (is_imputed == False).
"""

import numpy as np
import pandas as pd

from pipeline.config import BASELINE_START, BASELINE_END


def compute_zscore(
    gwsa_mm: pd.Series,
    is_imputed: pd.Series,
    baseline_start: str = BASELINE_START,
    baseline_end: str = BASELINE_END,
) -> pd.Series:
    """Z-score de gwsa_mm par rapport à la moyenne/écart-type de référence.

    Paramètres
    ----------
    gwsa_mm : pd.Series
        Série du proxy GWSA en mm, indexée par date mensuelle.
    is_imputed : pd.Series
        Booléen, True pour les mois interpolés.
    baseline_start, baseline_end : str
        Bornes de la période de référence (ex : '2004-01', '2009-12').

    Retourne
    --------
    pd.Series
        Z-score pour chaque mois. NaN si l'écart-type de référence est nul.
    """
    # --- Extraire les mois observés dans la fenêtre de référence ---
    mask_baseline = (
        (gwsa_mm.index >= baseline_start)
        & (gwsa_mm.index <= baseline_end)
        & (~is_imputed)
    )
    baseline_values = gwsa_mm.loc[mask_baseline]

    # --- Moyenne et écart-type de la référence ---
    mu = baseline_values.mean()
    sigma = baseline_values.std(ddof=1)  # écart-type échantillon

    # --- Calcul du z-score sur toute la série ---
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.nan, index=gwsa_mm.index, name="zscore")

    zscore = (gwsa_mm - mu) / sigma
    zscore.name = "zscore"
    return zscore


def compute_percentile_rank(
    gwsa_mm: pd.Series,
    is_imputed: pd.Series,
) -> pd.Series:
    """Rang percentile de chaque mois parmi tous les mois observés.

    Le percentile indique le pourcentage de mois observés dont la valeur
    est INFÉRIEURE ou ÉGALE à celle du mois courant.
    0 % ≈ le pire mois, 100 % ≈ le meilleur.

    Paramètres
    ----------
    gwsa_mm : pd.Series
        Série du proxy GWSA en mm.
    is_imputed : pd.Series
        Booléen, True pour les mois interpolés.

    Retourne
    --------
    pd.Series
        Rang percentile (0-100) pour chaque mois.
    """
    # --- Population de référence = tous les mois observés ---
    observed = gwsa_mm.loc[~is_imputed].dropna()
    n = len(observed)

    if n == 0:
        return pd.Series(np.nan, index=gwsa_mm.index, name="percentile_rank")

    # --- Pour chaque mois, compter combien de mois observés sont <= ---
    # On utilise np.searchsorted sur les valeurs triées
    sorted_obs = np.sort(observed.values)

    def _rank(value):
        if np.isnan(value):
            return np.nan
        # Nombre de mois observés <= value
        count = np.searchsorted(sorted_obs, value, side="right")
        return (count / n) * 100.0

    pct = gwsa_mm.apply(_rank)
    pct.name = "percentile_rank"
    return pct


def compute_anomaly_indicators(
    df: pd.DataFrame,
    value_col: str = "gwsa_mm",
    imputed_col: str = "is_imputed",
) -> pd.DataFrame:
    """Ajoute les colonnes zscore et percentile_rank au DataFrame.

    Paramètres
    ----------
    df : pd.DataFrame
        Doit contenir les colonnes value_col et imputed_col.
    value_col : str
        Nom de la colonne contenant la série GWSA (mm).
    imputed_col : str
        Nom de la colonne booléenne d'imputation.

    Retourne
    --------
    pd.DataFrame
        Copie du DataFrame avec colonnes 'zscore' et 'percentile_rank' ajoutées.
    """
    result = df.copy()
    result["zscore"] = compute_zscore(df[value_col], df[imputed_col])
    result["percentile_rank"] = compute_percentile_rank(
        df[value_col], df[imputed_col]
    )
    return result