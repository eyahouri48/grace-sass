# fichier : notebooks/11_sarima_ets_comparison.py
"""
Exploration — SARIMA + ETS + comparaison avec Prophet.

Ordres SARIMA guidés par l'ACF/PACF de la Tâche 8 :
  - ARIMA(2,1,0) — PACF significative aux lags 1 et 2
  - ARIMA(1,1,0) — variante plus simple
  - SARIMA(2,1,0)(1,0,0)₁₂ — avec terme saisonnier (ACF lag 12)
ETS Holt-Winters additif comme baseline.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pipeline.config import SERIES_PARQUET
from pipeline.forecast import (
    walk_forward_cv,
    compute_cv_metrics,
    compare_models,
)


# ── 1. Chargement ───────────────────────────────────────────────

print("Chargement du cache Parquet...")
df_cache = pd.read_parquet(SERIES_PARQUET)

gwsa_mm    = df_cache["gwsa_mm"]
is_imputed = df_cache["is_imputed"]

print(f"  Série : {len(gwsa_mm)} mois, {(~is_imputed).sum()} observés")


# ── 2. Walk-forward SARIMA — 3 variantes ────────────────────────

configs_sarima = {
    "ARIMA(1,1,0)": {"order": (1, 1, 0), "seasonal_order": (0, 0, 0, 12)},
    "ARIMA(2,1,0)": {"order": (2, 1, 0), "seasonal_order": (0, 0, 0, 12)},
    "SARIMA(2,1,0)(1,0,0)₁₂": {"order": (2, 1, 0), "seasonal_order": (1, 0, 0, 12)},
}

all_metrics = {}

# Résultats Prophet (de la Tâche 10 — recopiés des logs)
# ⚠ Remplacer par les vraies valeurs de ta Tâche 10
all_metrics["Prophet (saisonnier)"] = {"mae": 6.09, "rmse": 7.77, "n_obs": None}

for name, cfg in configs_sarima.items():
    print(f"\n--- Walk-forward {name} ---")
    cv_df = walk_forward_cv(
        gwsa_mm, is_imputed,
        model_type="sarima",
        order=cfg["order"],
        seasonal_order=cfg["seasonal_order"],
    )
    metrics = compute_cv_metrics(cv_df, observed_only=True)
    all_metrics[name] = metrics
    print(f"  MAE = {metrics['mae']:.2f} mm,  RMSE = {metrics['rmse']:.2f} mm  "
          f"({metrics['n_obs']} mois observés scorés)")


# ── 3. Walk-forward ETS (baseline) ──────────────────────────────

print("\n--- Walk-forward ETS (Holt-Winters additif) ---")
cv_ets = walk_forward_cv(
    gwsa_mm, is_imputed,
    model_type="ets",
)
metrics_ets = compute_cv_metrics(cv_ets, observed_only=True)
all_metrics["ETS Holt-Winters"] = metrics_ets
print(f"  MAE = {metrics_ets['mae']:.2f} mm,  RMSE = {metrics_ets['rmse']:.2f} mm  "
      f"({metrics_ets['n_obs']} mois observés scorés)")


# ── 4. Tableau comparatif ──────────────────────────────────────

print("\n" + "=" * 65)
print("COMPARAISON FINALE DES MODÈLES (mois observés uniquement)")
print("=" * 65)
comparison = compare_models(all_metrics)
print(comparison.to_string(index=False))

best_model_name = comparison.iloc[0]["modèle"]
print(f"\n→ Meilleur modèle (MAE) : {best_model_name}")


# ── 5. Graphique comparatif ────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

# Barres groupées MAE / RMSE
models = comparison["modèle"]
x = range(len(models))
width = 0.35

bars_mae = ax.bar([i - width/2 for i in x], comparison["mae_mm"],
                  width, label="MAE (mm)", color="#2c3e50")
bars_rmse = ax.bar([i + width/2 for i in x], comparison["rmse_mm"],
                   width, label="RMSE (mm)", color="#e74c3c", alpha=0.7)

ax.set_xticks(list(x))
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylabel("Erreur (mm)")
ax.set_title("Comparaison des modèles de prévision — validation croisée "
             "(mois observés uniquement)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Annoter les barres MAE avec la valeur
for bar in bars_mae:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("notebooks/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n Graphique sauvegardé : notebooks/model_comparison.png")