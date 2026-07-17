# fichier : notebooks/10_prophet_cv.py
"""
Exploration — Ajustement et validation Prophet sur gwsa_mm.

Teste deux variantes :
  A) yearly_seasonality=True  (tendance + cycle annuel)
  B) yearly_seasonality=False (tendance seule)

La validation croisée à origine glissante tranche.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pipeline.config import SERIES_PARQUET
from pipeline.forecast import (
    prepare_prophet_df,
    fit_prophet,
    cross_validate_prophet,
    score_observed_only,
    make_forecast,
)


# ── 1. Chargement des données ───────────────────────────────────

print("Chargement du cache Parquet...")
df_cache = pd.read_parquet(SERIES_PARQUET)

gwsa_mm    = df_cache["gwsa_mm"]
is_imputed = df_cache["is_imputed"]

print(f"  Série gwsa_mm : {len(gwsa_mm)} mois, "
      f"de {gwsa_mm.index[0]:%Y-%m} à {gwsa_mm.index[-1]:%Y-%m}")
print(f"  Mois imputés  : {is_imputed.sum()}")
print(f"  Mois observés : {(~is_imputed).sum()}")


# ── 2. Préparation des données ──────────────────────────────────

# Prophet reçoit la série SANS les mois imputés (il tolère les trous)
df_prophet = prepare_prophet_df(gwsa_mm, is_imputed, drop_imputed=True)
print(f"\nDataFrame Prophet : {len(df_prophet)} lignes (mois observés)")


# ── 3. Ajustement des deux variantes ────────────────────────────

print("\n--- Variante A : yearly_seasonality=True ---")
model_a = fit_prophet(df_prophet, yearly_seasonality=True)

print("\n--- Variante B : yearly_seasonality=False ---")
model_b = fit_prophet(df_prophet, yearly_seasonality=False)


# ── 4. Validation croisée à origine glissante ───────────────────
# ⚠️ C'est la partie la plus longue (~1-3 min par variante)

print("\nValidation croisée variante A (saisonnière)...")
cv_a = cross_validate_prophet(model_a)
metrics_a = score_observed_only(cv_a, is_imputed)
print(metrics_a[["horizon", "mae", "rmse"]].to_string())

print("\nValidation croisée variante B (non saisonnière)...")
cv_b = cross_validate_prophet(model_b)
metrics_b = score_observed_only(cv_b, is_imputed)
print(metrics_b[["horizon", "mae", "rmse"]].to_string())


# ── 5. Comparaison des deux variantes ───────────────────────────

# MAE moyenne sur tout l'horizon
mae_a = metrics_a["mae"].mean()
mae_b = metrics_b["mae"].mean()
rmse_a = metrics_a["rmse"].mean()
rmse_b = metrics_b["rmse"].mean()

print("\n" + "=" * 55)
print("COMPARAISON DES VARIANTES (moyennes sur tout l'horizon)")
print("=" * 55)
print(f"  Variante A (saisonnière)     : MAE = {mae_a:.2f} mm,  RMSE = {rmse_a:.2f} mm")
print(f"  Variante B (non saisonnière) : MAE = {mae_b:.2f} mm,  RMSE = {rmse_b:.2f} mm")

meilleure = "A (saisonnière)" if mae_a < mae_b else "B (non saisonnière)"
print(f"\n  → Meilleure variante (MAE) : {meilleure}")


# ── 6. Prévision finale avec le meilleur modèle ────────────────

best_model = model_a if mae_a < mae_b else model_b
forecast = make_forecast(best_model, horizon_months=24)

# Séparer historique et futur pour le graphique
last_obs_date = gwsa_mm.index[-1]
forecast_future = forecast[forecast["ds"] > last_obs_date]

print(f"\nPrévision sur 24 mois : de {forecast_future['ds'].iloc[0]:%Y-%m} "
      f"à {forecast_future['ds'].iloc[-1]:%Y-%m}")
print(f"  yhat finale : {forecast_future['yhat'].iloc[-1]:.1f} mm")
print(f"  IC 80%      : [{forecast_future['yhat_lower'].iloc[-1]:.1f}, "
      f"{forecast_future['yhat_upper'].iloc[-1]:.1f}] mm")


# ── 7. Graphiques ───────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# --- Graphique 1 : série observée + prévision Prophet ---
ax1 = axes[0]

# Série observée (mois réels en bleu, imputés en gris clair)
obs_mask = ~is_imputed
ax1.plot(gwsa_mm.index[obs_mask], gwsa_mm[obs_mask],
         "o", ms=2, color="#2c3e50", label="Observé (GRACE)", zorder=3)
ax1.plot(gwsa_mm.index[is_imputed], gwsa_mm[is_imputed],
         "o", ms=2, color="#bdc3c7", label="Imputé (interpolé)", zorder=2)

# Lacune 2017-2018 — zone grisée
ax1.axvspan(pd.Timestamp("2017-07-01"), pd.Timestamp("2018-06-01"),
            alpha=0.15, color="gray", label="Lacune inter-missions")

# Prévision future — ligne + bande d'incertitude
ax1.plot(forecast_future["ds"], forecast_future["yhat"],
         color="#e74c3c", linewidth=2, label="Prévision Prophet (24 mois)")
ax1.fill_between(
    forecast_future["ds"],
    forecast_future["yhat_lower"],
    forecast_future["yhat_upper"],
    alpha=0.2, color="#e74c3c", label="IC 80%",
)

ax1.set_ylabel("GWSA proxy (mm)")
ax1.set_title(
    f"Proxy GWSA du SASS — observation + prévision Prophet ({meilleure})",
    fontsize=13,
)
ax1.legend(loc="lower left", fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Graphique 2 : MAE en fonction de l'horizon ---
ax2 = axes[1]

# Convertir l'horizon de timedelta en jours pour un axe lisible
best_metrics = metrics_a if mae_a < mae_b else metrics_b
horizon_days = best_metrics["horizon"].dt.days
ax2.plot(horizon_days, best_metrics["mae"], "o-", color="#2c3e50", label="MAE")
ax2.plot(horizon_days, best_metrics["rmse"], "s--", color="#e74c3c", label="RMSE")

ax2.set_xlabel("Horizon de prévision (jours)")
ax2.set_ylabel("Erreur (mm)")
ax2.set_title("Performance de la prévision en fonction de l'horizon "
              "(mois observés uniquement)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("notebooks/prophet_forecast.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n Graphique sauvegardé : notebooks/prophet_forecast.png")