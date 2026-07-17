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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

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



# ── 5. SARIMA ────────────────────────────────────────────────────

def fit_sarima(
    gwsa_mm: pd.Series,
    order: tuple = (2, 1, 0),
    seasonal_order: tuple = (0, 0, 0, 12),
) -> object:
    """Ajuster un modèle SARIMA sur la série gwsa_mm (continue, interpolée).

    Parameters
    ----------
    gwsa_mm : pd.Series
        Série GWSA mensuelle CONTINUE (mois imputés inclus).
        SARIMA ne tolère pas les trous — c'est la série interpolée.
    order : tuple
        (p, d, q) — ordres non saisonniers.
    seasonal_order : tuple
        (P, D, Q, s) — ordres saisonniers. (0,0,0,12) = pas de saisonnalité.

    Returns
    -------
    SARIMAXResultsWrapper
        Modèle ajusté avec .forecast() et .get_forecast() disponibles.
    """
    model = SARIMAX(
        gwsa_mm,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,   # évite des erreurs numériques
        enforce_invertibility=False,  # idem
    )
    results = model.fit(disp=False)  # disp=False : pas de log d'optimisation
    return results


def walk_forward_cv(
    gwsa_mm: pd.Series,
    is_imputed: pd.Series,
    model_type: str = "sarima",
    order: tuple = (2, 1, 0),
    seasonal_order: tuple = (0, 0, 0, 12),
    initial_months: int = 96,
    step_months: int = 12,
    horizon_months: int = 24,
) -> pd.DataFrame:
    """Validation croisée walk-forward pour SARIMA ou ETS.

    Reproduit la même logique que la CV Prophet : fenêtre expansive,
    avance d'un an par pli, horizon de 24 mois, scoring sur mois
    observés uniquement.

    Parameters
    ----------
    gwsa_mm : pd.Series
        Série continue (interpolée) avec DatetimeIndex mensuel.
    is_imputed : pd.Series
        Booléen — True pour les mois interpolés.
    model_type : str
        "sarima" ou "ets".
    order, seasonal_order : tuple
        Ordres SARIMA (ignorés si model_type="ets").
    initial_months : int
        Fenêtre d'entraînement minimale (défaut 96 ≈ 8 ans, comme Prophet).
    step_months : int
        Avance entre les plis (défaut 12 ≈ 1 an).
    horizon_months : int
        Horizon de prévision par pli (défaut 24).

    Returns
    -------
    pd.DataFrame
        Colonnes : cutoff, ds, y_true, y_pred, is_obs.
        is_obs = True si le mois est réellement observé (pas imputé).
    """
    results_list = []
    n = len(gwsa_mm)

    # Générer les coupures (cutoffs)
    cutoff_indices = list(range(initial_months, n - horizon_months, step_months))

    for cut_idx in cutoff_indices:
        # Fenêtre d'entraînement : du début jusqu'au cutoff
        train = gwsa_mm.iloc[:cut_idx].copy()
        train.index.freq = "MS"  # forcer la fréquence mensuelle (supprime le warning statsmodels)
        # Fenêtre de test : du cutoff au cutoff + horizon
        test = gwsa_mm.iloc[cut_idx : cut_idx + horizon_months]

        if len(test) == 0:
            continue

        # Ajuster le modèle sur la fenêtre d'entraînement
        try:
            if model_type == "sarima":
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False)
                pred = fit.forecast(steps=len(test))
            elif model_type == "ets":
                model = ExponentialSmoothing(
                    train,
                    trend="add",
                    seasonal=None,        # pas de saisonnalité (bassin fossile)
                    initialization_method="estimated",
                )
                fit = model.fit(optimized=True)
                pred = fit.forecast(steps=len(test))
            else:
                raise ValueError(f"model_type inconnu : {model_type}")
        except Exception as e:
            # Si un pli échoue (convergence...), on le saute
            print(f"  ⚠ Pli cutoff={train.index[-1]:%Y-%m} échoué : {e}")
            continue

        # Collecter les résultats
        for i, (date, y_true) in enumerate(test.items()):
            results_list.append({
                "cutoff": train.index[-1],
                "ds": date,
                "y_true": y_true,
                "y_pred": pred.iloc[i],
                "is_obs": not is_imputed.loc[date],
            })

    return pd.DataFrame(results_list)


def compute_cv_metrics(
    cv_df: pd.DataFrame,
    observed_only: bool = True,
) -> dict:
    """Calculer MAE et RMSE à partir du DataFrame walk-forward.

    Parameters
    ----------
    cv_df : pd.DataFrame
        Sortie de walk_forward_cv(). Colonnes : y_true, y_pred, is_obs.
    observed_only : bool
        Si True (défaut), ne score que sur les mois observés (is_obs=True).

    Returns
    -------
    dict
        {"mae": float, "rmse": float, "n_obs": int}
    """
    if observed_only:
        df = cv_df[cv_df["is_obs"]].copy()
    else:
        df = cv_df.copy()

    errors = df["y_true"] - df["y_pred"]
    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())

    return {"mae": mae, "rmse": rmse, "n_obs": len(df)}


# ── 6. ETS / Holt-Winters ───────────────────────────────────────

def fit_ets(
    gwsa_mm: pd.Series,
    trend: str = "add",
    seasonal: str = None,
) -> object:
    """Ajuster un modèle Holt-Winters (ETS) additif.

    Parameters
    ----------
    gwsa_mm : pd.Series
        Série continue (interpolée).
    trend : str
        "add" pour tendance additive (défaut).
    seasonal : str or None
        None = pas de saisonnalité (recommandé sur le SASS).

    Returns
    -------
    HoltWintersResultsWrapper
        Modèle ajusté.
    """
    model = ExponentialSmoothing(
        gwsa_mm,
        trend=trend,
        seasonal=seasonal,
        initialization_method="estimated",
    )
    return model.fit(optimized=True)


# ── 7. Comparaison des modèles ──────────────────────────────────

def compare_models(metrics_dict: dict) -> pd.DataFrame:
    """Créer un tableau comparatif des modèles.

    Parameters
    ----------
    metrics_dict : dict
        Clé = nom du modèle, valeur = dict {"mae": ..., "rmse": ..., "n_obs": ...}
        Ex : {"Prophet (saisonnier)": {"mae": 6.09, ...}, "SARIMA(2,1,0)": {...}}

    Returns
    -------
    pd.DataFrame
        Tableau trié par MAE croissante, colonnes : modèle, mae, rmse, n_obs.
    """
    rows = []
    for name, m in metrics_dict.items():
        rows.append({"modèle": name, "mae_mm": m["mae"], "rmse_mm": m["rmse"], "n_obs": m["n_obs"]})

    df = pd.DataFrame(rows).sort_values("mae_mm").reset_index(drop=True)
    return df