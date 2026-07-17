# fichier : pipeline/forecast.py
"""
Prévision de la série GWSA par Prophet (et SARIMA — ajouté à la Tâche 11).

Ce module contient :
- La préparation des données pour Prophet
- L'ajustement du modèle (avec/sans saisonnalité)
- La validation croisée à origine glissante (rolling-origin CV)
- Le filtrage des mois imputés dans le scoring
- La production de la prévision finale avec intervalle d'incertitude
"""

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from pipeline.config import (
    CV_INITIAL,
    CV_PERIOD,
    CV_HORIZON,
    PROPHET_CHANGEPOINT_PRIOR_SCALE,
    FORECAST_HORIZON_MONTHS,
)


# ── 1. Préparation des données ──────────────────────────────────

def prepare_prophet_df(
    gwsa_mm: pd.Series,
    is_imputed: pd.Series,
    drop_imputed: bool = True,
) -> pd.DataFrame:
    """Convertir la série gwsa_mm en DataFrame Prophet (colonnes ds, y).

    Parameters
    ----------
    gwsa_mm : pd.Series
        Anomalie de stockage souterrain (mm), index DatetimeIndex.
    is_imputed : pd.Series
        Booléen — True pour les mois interpolés (lacune 2017-2018 etc.).
    drop_imputed : bool
        Si True (défaut), on retire les mois imputés du DataFrame.
        Prophet tolère les trous — il travaille sur les mois observés.

    Returns
    -------
    pd.DataFrame
        Colonnes 'ds' (datetime) et 'y' (float), sans les mois imputés.
    """
    # Construire le DataFrame au format Prophet
    df = pd.DataFrame({"ds": gwsa_mm.index, "y": gwsa_mm.values})

    if drop_imputed:
        # Garder uniquement les mois réellement observés par GRACE
        masque_observe = ~is_imputed.values
        df = df.loc[masque_observe].reset_index(drop=True)

    return df


# ── 2. Ajustement du modèle ─────────────────────────────────────

def fit_prophet(
    df: pd.DataFrame,
    yearly_seasonality: bool = True,
    changepoint_prior_scale: float = PROPHET_CHANGEPOINT_PRIOR_SCALE,
) -> Prophet:
    """Ajuster un modèle Prophet sur les données préparées.

    Parameters
    ----------
    df : pd.DataFrame
        Format Prophet (colonnes ds, y) — typiquement la sortie de
        prepare_prophet_df().
    yearly_seasonality : bool
        Active ou désactive la saisonnalité annuelle. On teste les deux
        et on laisse la CV trancher (§7.1).
    changepoint_prior_scale : float
        Curseur rigidité de la tendance (§7.1). Plus c'est bas, plus
        la tendance est rigide ; plus c'est haut, plus elle épouse les
        variations locales.

    Returns
    -------
    Prophet
        Modèle ajusté, prêt pour .predict() ou cross_validation().
    """
    # Créer le modèle — on désactive les logs Prophet (bavards par défaut)
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=False,   # données mensuelles — pas de cycle hebdo
        daily_seasonality=False,    # idem
        changepoint_prior_scale=changepoint_prior_scale,
    )

    # Ajuster sur les données observées
    model.fit(df)

    return model


# ── 3. Validation croisée à origine glissante ───────────────────

def cross_validate_prophet(
    model: Prophet,
    initial: str = CV_INITIAL,
    period: str = CV_PERIOD,
    horizon: str = CV_HORIZON,
) -> pd.DataFrame:
    """Exécuter la validation croisée à origine glissante de Prophet.

    Chaque pli :
    - Entraîne sur les données jusqu'à la date de coupure
    - Prévoit sur l'horizon (jusqu'à 24 mois)
    - Compare la prévision (yhat) à l'observation réelle (y)

    Parameters
    ----------
    model : Prophet
        Modèle déjà ajusté via fit_prophet().
    initial, period, horizon : str
        Paramètres de la CV (§7.2 du cahier des charges).

    Returns
    -------
    pd.DataFrame
        Colonnes : ds, yhat, yhat_lower, yhat_upper, y, cutoff.
        Chaque ligne = une prévision pour un mois dans un pli donné.
    """
    cv_results = cross_validation(
        model,
        initial=initial,
        period=period,
        horizon=horizon,
    )
    return cv_results


def score_observed_only(
    cv_results: pd.DataFrame,
    is_imputed: pd.Series,
) -> pd.DataFrame:
    """Filtrer les mois imputés et calculer MAE/RMSE sur les mois observés.

    C'est ICI que la règle §7.2 s'applique : on ne score jamais contre
    une valeur qu'on a soi-même interpolée.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Sortie brute de cross_validate_prophet() — contient ds, y, yhat.
    is_imputed : pd.Series
        Booléen indexé par DatetimeIndex. True = mois interpolé.

    Returns
    -------
    pd.DataFrame
        Métriques Prophet (horizon, mae, rmse…) calculées sur les mois
        observés uniquement.
    """
    # Normaliser les dates de cv_results au premier du mois
    # (Prophet peut décaler légèrement les timestamps)
    cv_clean = cv_results.copy()
    cv_clean["ds_month"] = cv_clean["ds"].dt.to_period("M")

    # Construire un set des mois imputés pour un filtrage rapide
    imputed_months = set(
        is_imputed[is_imputed].index.to_period("M")
    )

    # Retirer les lignes dont le mois de test est imputé
    masque = ~cv_clean["ds_month"].isin(imputed_months)
    cv_observed = cv_clean.loc[masque].drop(columns=["ds_month"])

    # Calculer les métriques sur les mois observés seulement
    metrics = performance_metrics(cv_observed)

    return metrics


# ── 4. Prévision finale ─────────────────────────────────────────

def make_forecast(
    model: Prophet,
    horizon_months: int = FORECAST_HORIZON_MONTHS,
) -> pd.DataFrame:
    """Produire la prévision future avec intervalle d'incertitude.

    Parameters
    ----------
    model : Prophet
        Modèle ajusté.
    horizon_months : int
        Nombre de mois à prévoir (défaut : 24 = horizon validé).

    Returns
    -------
    pd.DataFrame
        Colonnes clés : ds, yhat, yhat_lower, yhat_upper, trend.
        yhat_lower/upper = intervalle d'incertitude à 80% (défaut Prophet).
    """
    # Créer le DataFrame de dates futures
    future = model.make_future_dataframe(periods=horizon_months, freq="MS")

    # Produire la prévision (historique reconstruit + futur)
    forecast = model.predict(future)

    return forecast