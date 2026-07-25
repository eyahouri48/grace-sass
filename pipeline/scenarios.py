# fichier : pipeline/scenarios.py
"""
Cadrage par scénarios — horizon validé vs. extrapolation.

Produit une prévision Prophet sur deux horizons (validé ~24 mois,
extrapolation ~60 mois) avec étiquetage explicite de chaque zone
et métadonnées de confiance pour le dashboard et le rapport.
"""

import pandas as pd
from prophet import Prophet

from pipeline.config import (
    FORECAST_HORIZON_MONTHS,
    SCENARIO_HORIZON_MONTHS,
    PROPHET_CHANGEPOINT_PRIOR_SCALE,
)


def build_scenarios(
    gwsa_mm: pd.Series,
    validated_mae_mm: float,
    forecast_horizon: int = FORECAST_HORIZON_MONTHS,
    scenario_horizon: int = SCENARIO_HORIZON_MONTHS,
    changepoint_prior_scale: float = PROPHET_CHANGEPOINT_PRIOR_SCALE,
    yearly_seasonality: bool = True,
) -> dict:
    """
    Produit la prévision Prophet sur l'horizon de scénario complet
    et la découpe en zone validée / zone d'extrapolation.

    Parameters
    ----------
    gwsa_mm : pd.Series
        Série mensuelle gwsa_mm (index DatetimeIndex, en mm).
        Peut contenir des NaN — Prophet les tolère.
    validated_mae_mm : float
        MAE issue de la CV à origine glissante (mm), pour documentation.
    forecast_horizon : int
        Horizon validé en mois (par défaut 24).
    scenario_horizon : int
        Horizon total du scénario en mois (par défaut 60).
    changepoint_prior_scale : float
        Paramètre Prophet (flexibilité de la tendance).
    yearly_seasonality : bool
        Activer la saisonnalité annuelle (True par défaut — la CV a tranché).

    Returns
    -------
    dict avec les clés :
        "forecast_df"      : DataFrame complet (ds, yhat, yhat_lower, yhat_upper, zone)
        "last_obs_date"     : date de la dernière observation
        "validated_horizon" : int (mois)
        "scenario_horizon"  : int (mois)
        "validated_mae_mm"  : float
        "cutoff_date"       : date de séparation validé / extrapolation
        "warnings"          : dict de textes d'avertissement (EN/FR)
    """
    # --- Ajuster Prophet sur toute la série observée ---
    df_prophet = (
        gwsa_mm
        .dropna()
        .reset_index()
        .rename(columns={gwsa_mm.index.name or "index": "ds", gwsa_mm.name or 0: "y"})
    )
    # S'assurer que la colonne cible s'appelle bien "y"
    if "y" not in df_prophet.columns:
        df_prophet = df_prophet.rename(columns={df_prophet.columns[-1]: "y"})

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    model.fit(df_prophet)

    # --- Générer la prévision sur l'horizon total (scénario) ---
    future = model.make_future_dataframe(periods=scenario_horizon, freq="MS")
    forecast = model.predict(future)

    # --- Découper en zones ---
    last_obs_date = df_prophet["ds"].max()
    cutoff_date = last_obs_date + pd.DateOffset(months=forecast_horizon)

    # Ne garder que la partie future (après la dernière observation)
    forecast_future = forecast[forecast["ds"] > last_obs_date].copy()

    forecast_future["zone"] = "extrapolation"
    forecast_future.loc[
        forecast_future["ds"] <= cutoff_date, "zone"
    ] = "validated"

    # --- Colonnes utiles pour le dashboard ---
    result_df = forecast_future[
        ["ds", "yhat", "yhat_lower", "yhat_upper", "zone"]
    ].reset_index(drop=True)

    # --- Textes d'avertissement bilingues ---
    warnings = _build_warning_texts(
        forecast_horizon, scenario_horizon, validated_mae_mm
    )

    return {
        "forecast_df": result_df,
        "last_obs_date": last_obs_date,
        "validated_horizon": forecast_horizon,
        "scenario_horizon": scenario_horizon,
        "validated_mae_mm": validated_mae_mm,
        "cutoff_date": cutoff_date,
        "warnings": warnings,
    }


def _build_warning_texts(
    forecast_horizon: int,
    scenario_horizon: int,
    validated_mae_mm: float,
) -> dict:
    """
    Construit les textes d'avertissement pour le rapport et le dashboard.
    """
    return {
        "en": {
            "validated": (
                f"Validated forecast (0–{forecast_horizon} months): "
                f"rolling-origin CV confirms skill with MAE ≈ {validated_mae_mm:.1f} mm."
            ),
            "extrapolation": (
                f"Scenario extrapolation ({forecast_horizon}–{scenario_horizon} months): "
                f"trend continuation under current conditions — NOT a validated forecast. "
                f"Confidence interval widens; cannot anticipate policy, pumping, "
                f"or climate shifts."
            ),
        },
        "fr": {
            "validated": (
                f"Prévision validée (0–{forecast_horizon} mois) : "
                f"la CV à origine glissante confirme la performance avec "
                f"MAE ≈ {validated_mae_mm:.1f} mm."
            ),
            "extrapolation": (
                f"Extrapolation de scénario ({forecast_horizon}–{scenario_horizon} mois) : "
                f"continuation de tendance sous conditions actuelles — "
                f"PAS une prévision validée. L'intervalle de confiance s'élargit ; "
                f"ne peut anticiper les changements de politique, de prélèvement "
                f"ou de climat."
            ),
        },
    }


def get_scenario_summary(scenarios: dict) -> pd.DataFrame:
    """
    Produit un tableau de synthèse lisible : valeur prévue et IC
    à des jalons clés (12, 24, 36, 48, 60 mois).

    Returns
    -------
    DataFrame avec colonnes : horizon_months, date, yhat_mm, lower_mm, upper_mm, zone
    """
    df = scenarios["forecast_df"]
    last_obs = scenarios["last_obs_date"]
    milestones = [12, 24, 36, 48, 60]

    rows = []
    for m in milestones:
        if m > scenarios["scenario_horizon"]:
            continue
        target_date = last_obs + pd.DateOffset(months=m)
        # Trouver la ligne la plus proche de la date cible
        idx = (df["ds"] - target_date).abs().idxmin()
        row = df.loc[idx]
        rows.append(
            {
                "horizon_months": m,
                "date": row["ds"],
                "yhat_mm": round(row["yhat"], 1),
                "lower_mm": round(row["yhat_lower"], 1),
                "upper_mm": round(row["yhat_upper"], 1),
                "zone": row["zone"],
            }
        )

    return pd.DataFrame(rows)


def build_multi_scenarios(
    gwsa_mm: pd.Series,
    is_imputed: pd.Series,
    validated_mae_mm: float,
    forecast_horizon: int = FORECAST_HORIZON_MONTHS,
    scenario_horizon: int = SCENARIO_HORIZON_MONTHS,
) -> dict:
    """
    Construit 3 scénarios : central (Prophet), sécheresse, gestion renforcée.

    Le scénario central est le forecast Prophet standard.
    Les scénarios alternatifs appliquent un décalage linéaire basé sur
    la pente de Sen (±50 %) — ce sont des what-if, PAS des prévisions.

    Returns
    -------
    dict avec clés :
        "central_df", "dry_df", "wet_df" : DataFrames (ds, yhat, yhat_lower, yhat_upper, zone)
        "last_obs_date", "cutoff_date", "sen_slope_mm_yr"
    """
    from pipeline.trend import mann_kendall_sen

    central = build_scenarios(
        gwsa_mm=gwsa_mm,
        validated_mae_mm=validated_mae_mm,
        forecast_horizon=forecast_horizon,
        scenario_horizon=scenario_horizon,
    )

    sen_result = mann_kendall_sen(gwsa_mm, is_imputed)
    sen_slope = sen_result["sen_slope_mm_yr"]

    forecast_df = central["forecast_df"].copy()
    last_obs_date = central["last_obs_date"]

    t_years = (forecast_df["ds"] - last_obs_date).dt.days / 365.25
    perturbation_factor = 0.5

    # Sécheresse : accélération de la baisse (sen_slope < 0 → delta < 0)
    delta_dry = sen_slope * perturbation_factor * t_years
    dry_df = forecast_df.copy()
    dry_df["yhat"] = forecast_df["yhat"] + delta_dry
    dry_df["yhat_upper"] = forecast_df["yhat_upper"] + delta_dry
    dry_df["yhat_lower"] = forecast_df["yhat_lower"] + delta_dry

    # Gestion renforcée : ralentissement de la baisse
    delta_wet = -sen_slope * perturbation_factor * t_years
    wet_df = forecast_df.copy()
    wet_df["yhat"] = forecast_df["yhat"] + delta_wet
    wet_df["yhat_upper"] = forecast_df["yhat_upper"] + delta_wet
    wet_df["yhat_lower"] = forecast_df["yhat_lower"] + delta_wet

    cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "zone"]
    return {
        "central_df": forecast_df[cols].copy(),
        "dry_df": dry_df[cols].copy(),
        "wet_df": wet_df[cols].copy(),
        "last_obs_date": last_obs_date,
        "cutoff_date": central["cutoff_date"],
        "sen_slope_mm_yr": sen_slope,
        "validated_mae_mm": validated_mae_mm,
    }